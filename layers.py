import logging
from argparse import Namespace
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Function

from utils import clamp_ste, round_ste, floor_ste


class MultiLevelFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: Tensor, th: float, L: int, sigma: float = 0.8) -> Tensor:
        r"""

        :param ctx:
        :param input:
        :param th: 初值应该为量化的时候的 scale
        :param L:
        :param sigma:
        :return:
        """
        k = (input / th).floor().clamp(0, L)
        out = k * th
        ctx.save_for_backward(input, th)
        ctx.L = L
        ctx.sigma = sigma
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, th = ctx.saved_tensors
        L = ctx.L
        sigma = ctx.sigma

        # 计算 grad_input
        grad_input = torch.zeros_like(input)
        for k in range(1, L + 1):
            center = k * th
            surrogate = torch.sigmoid((input - center) / sigma)
            grad_surrogate = surrogate * (1 - surrogate) / sigma
            grad_input += grad_surrogate
        grad_input = grad_output * grad_input

        # 计算 grad_th
        grad_th = torch.zeros_like(th)
        for k in range(1, L + 1):
            center = k * th
            surrogate = torch.sigmoid((input - center) / sigma)
            # 推导 grad_th：d(sigmoid)/d(center) * d(center)/d(th)
            grad_surrogate = surrogate * \
                             (1 - surrogate) / sigma  # d(sigmoid)/d(center)
            # d(sigmoid)/d(center) 对 center 求导是负的
            grad_center = -grad_surrogate
            # d(center)/d(th) = k，沿着 input 维度求和
            grad_th += (grad_center * k).sum()
        grad_th = grad_output * grad_th  # 应用链式法则

        return grad_input, grad_th, None, None


class MLSWATFunction(Function):
    r"""
    == 用于动态多阈值的脉冲释放函数。 ==
    动态适应性：无论阈值间距是否相等，自动调整邻域范围。
    梯度传播合理性：仅在输入值接近阈值时传播梯度，避免离散跳跃导致的训练不稳定。
    兼容性：在等间距场景下与原函数行为一致（如 [4th, 3th, 2th, th] → 范围 [0.5th, 4.5th]）。
    """

    @staticmethod
    def forward(ctx: Any, inputTensor, threshes: Tensor):
        r"""

        :param ctx:
        :param inputTensor:
        :type inputTensor:Tensor
        :param threshes: 一维张量。多个阈值从大到小排序的。
        :type threshes:Tensor
        :return:
        :rtype:Tensor
        """
        # region ---多阈值互斥累加---
        # 掩码张量。用来排除被比较过的阈值。
        rem = torch.ones_like(inputTensor, dtype=inputTensor.dtype)
        # 初始化输出结果。
        out = torch.zeros_like(inputTensor, dtype=inputTensor.dtype)
        # 依次和每个阈值比较。
        for thresh in threshes:
            # 当前阈值的匹配掩码
            mask = (inputTensor >= thresh).float()
            # 排除已被更大阈值匹配的部分
            exclusive = mask * rem
            # 在当前迭代累加当前阈值
            out = out + exclusive * thresh
            # 更新剩余掩码，这些位置将不再被后续（更小的）阈值匹配
            rem = rem * (1.0 - mask)
        # endregion

        # region ——— 构造 backward 用的 tmp mask ———
        # 计算半步长：smallest = threshes[-1], second_smallest = threshes[-2]
        if threshes.numel() > 1:
            halfLower = (threshes[-2] - threshes[-1]) * 0.5
            halfUpper = (threshes[0] - threshes[1]) * 0.5
        else:
            # 只有一个阈值时，范围就是 [thresh/2, thresh*1.5]
            halfLower = threshes[0] * 0.5
            halfUpper = halfLower

        lower = threshes[-1] - halfLower
        upper = threshes[0] + halfUpper

        backward_mask = (
                (inputTensor.detach() >= lower) &
                (inputTensor.detach() <= upper)
        ).float()

        # 保存所有 backward 需要的张量
        ctx.save_for_backward(inputTensor, threshes, backward_mask)
        return out

    @staticmethod
    def backward(ctx, gradOutput: Tensor):
        r"""
        grad_output: 与 forward 返回的 out 同形，表示 dL/dout
        需要返回
        :param ctx:
        :param gradOutput:
        :type gradOutput: Tensor
        :return: (dL/dinputTensor, dL/dthreshes)
        :rtype: Tuple[Tensor, Tensor]
        """
        inputTensor, threshes, backward_mask = ctx.saved_tensors

        # ——— 输入梯度：原地屏蔽 = grad_output * mask ———
        gradInput = gradOutput * backward_mask

        # ——— 阈值梯度：按互斥逻辑直接累加到 grad_threshes ———
        rem = torch.ones_like(inputTensor, dtype=inputTensor.dtype)
        gradThreshes = torch.zeros_like(threshes)

        # 对每个阈值，梯度 ∂out/∂t_i = exclusive_i
        # 所以 dL/dt_i = sum( dL/dout * exclusive_i )
        for i, thresh in enumerate(threshes):
            mask = (inputTensor >= thresh).float()
            exclusive = mask * rem
            # 累加所有元素位置的梯度
            gradThreshes[i] = (gradOutput * exclusive).sum()
            rem = rem * (1.0 - mask)

        return gradInput, gradThreshes


class MLSWATNeuron(nn.Module):
    r"""
    == Multi-level Spiking with Adaptive Threshes ==

    需要注意。在定义阈值的时候，阈值是一个长度为 T 的列表。
    其中第一个元素为最低的阈值，后续的值代表的意思为：
    第 t+1 个元素的值，代表的是第 t+1 个阈值比第 t 个阈值大的部分。

    # TODO阈值的更新需要注意，在优化器执行完 step() 之后，需要手动裁切。比如：

    通过 threshes 属性，对更新后的阈值实时投影，保持其偏序关系。
    """

    def __init__(self,
                 args: Namespace,
                 scale: Tensor,
                 zero_point: Tensor,
                 nameLogger: str) -> None:
        r"""

        :param args:
        :type args:Namespace
        :param scale:
        :type scale: Tensor
        :param zero_point:
        :type zero_point:Tensor
        :param gamma
        :param nameLogger:
        :type nameLogger:str
        """
        super().__init__()
        self.logger = logging.getLogger(nameLogger)

        if args.ifLearnThreshInMLSWA:
            self.logger.info(f"指定了需要学习阈值。")
        else:
            self.logger.info(f"使用固定阈值。")

        self.T = args.T
        # self.L = 2 ** args.bitsForQuant // self.T  # SNN中每个时刻的量化范围应该是ANN中的量化范围再除以 latency T
        self.L = args.L

        # region 对阈值的设置。
        if args.threshes is not None:  # TODO 需要在 arguments 中单独添加该参数，区别于 args.thresh .
            # 指定阈值
            if not isinstance(args.threshes, Sequence):
                errorMsg = ValueError(f"args.threshes 要么为None，要么为浮点列表")

            elif len(args.threshes) != self.L:
                errorMsg = ValueError(
                    f"阈值的数量 len(args.threshes) 应该等于每个时刻可能发送的最多脉冲数 self.L -{self.L}。")
            else:
                errorMsg = None
                self.raw_threshes = nn.Parameter(torch.tensor(args.threshes),
                                                 requires_grad=args.ifLearnThreshInMLSWA)
        else:
            # 从 QANN 继承 scale 作为阈值。
            if scale is None:
                errorMsg = ValueError(f"当不显示指定阈值的时候，应该传入量化的 scale 用来定义阈值。")
            elif len(scale.shape) != 1:
                errorMsg = ValueError(f"选择从 QANN 继承 scale 的时候，scale 的必须是1维张量。")
            elif len(scale) != self.L:
                errorMsg = ValueError(f"选择从 QANN 继承 scale 的时候，scale 的元素个数 {len(scale)} 必须等于"
                                      f" 每个时刻可能发送的最多脉冲数 self.L - {self.L}。")
            else:
                errorMsg = None
                self.raw_threshes = nn.Parameter(scale,
                                                 requires_grad=args.ifLearnThreshInMLSWA)
        if errorMsg is not None:
            self.logger.error(errorMsg)
            raise errorMsg
        if not (self.threshes > 0).all().item():
            errorMsg = ValueError(f"阈值中的每个元素应该都是正数！")
            self.logger.info(errorMsg)
            raise errorMsg
        # endregion

        # region zero_point的设置
        if zero_point is not None:
            self.zero_point = nn.Parameter(zero_point,
                                           requires_grad=args.ifLearnZPinMSWATInMLSWA)
        else:
            errorMsg = ValueError(f"必须传入 zero_point，否则不能模拟负数激活值。")
            self.logger.info(errorMsg)
            raise errorMsg
        # endregion

        self.initial_mem = args.initial_mem * args.thresh
        self.v: Tensor = None

    def forward(self, inputTensor: Tensor) -> Tensor:
        r"""

        :param inputTensor:
        :type inputTensor:Tensor
        :return:
        :rtype: Tensor
        """
        threshes = self.threshes
        # 辅助电流。用以保证电压以及发射率可以为负数。脉冲数量始终为非负数。

        # TODO 这里的 threshes 是一维向量，为导致currentAuxiliary为相邻。而非标量。
        #  而我们在逼近 Sigmoid的 过程中，不需要考虑等价性，只需要维持 currentAuxiliary 为标量。所以不乘以threshes。
        # currentAuxiliary = threshes * self.zero_point / self.T
        currentAuxiliary = self.zero_point / self.T
        self.v = torch.ones_like(inputTensor[0]) * self.initial_mem
        spikesOut = list()
        for idxStep, t in enumerate(range(self.T)):
            self.v = self.v.detach() + inputTensor[t] + currentAuxiliary
            spikesCurr = MLSWATFunction.apply(self.v,
                                              threshes)
            self.v = self.v - spikesCurr.detach()
            spikesOut.append(spikesCurr - currentAuxiliary)
        return torch.stack(spikesOut, dim=0)

    @property
    def threshes(self) -> Tensor:
        r"""
        softplus(x)=ln(1+e^x)，确保输出 > 0
        作用是保证实际上的 threshes 中的元素都为正数。
        并且计算真实阈值，保证各个阈值的偏序关系。
        :return: 返回满足偏序关系且从大到小排序的阈值序列。
        :rtype: Tensor
        """
        # 通过投影，确保每个增量严格 > 0。
        increments = F.softplus(self.raw_threshes)  # Tensor, requires_grad=True
        # 计算真实阈值。
        real_threshes = torch.cumsum(increments, dim=0)  # 前缀和
        # 对从小到大的阈值序列，颠倒为从大到小排序。
        real_threshes = torch.flip(real_threshes, dims=(0,))  # 反转顺序
        return real_threshes


class LMHTNeuron(nn.Module):
    r"""
    == 根据 PrefixQuant 和 LMHT 实现的整数倍脉冲神经元。==
    """

    def __init__(self,
                 args: Namespace,
                 scale: Tensor,
                 zero_point: Tensor,
                 sigma=0.8):
        r"""
        :param args:
        :type args: Namespace
        :param scale: TODO 形状需要确定：(1,1) or () 量化的时候使用 scale.data，作为脉冲阈值。脉冲神经元不用自己定义脉冲阈值。
        :type scale: Tensor
        :param zero_point: TODO 形状需要确定(1,1) or ()。量化的时候计算的 zero_point.data。
        :type zero_point:Tensor
        :param sigma: YouRw 写的反传中代理梯度用到的参数。
        :type sigma:float
        """
        super(LMHTNeuron, self).__init__()
        if args.Thresh is not None:
            self.v_threshold = nn.Parameter(torch.tensor([args.Thresh]),
                                            requires_grad=True)  # 设置为可训练参数
        else:
            self.v_threshold = scale

        self.v: Tensor  # 初始电压。

        self.initial_mem = args.initial_mem * args.Thresh

        self.T = args.T
        self.L = 2 ** args.bitsForQuant // self.T  # SNN中每个时刻的量化范围应该是ANN中的量化范围再除以 latency T

        self.scale = nn.Parameter(scale,
                                  requires_grad=args.ifLearnScaleInSpike)
        self.zero_point = nn.Parameter(zero_point,
                                       requires_grad=args.ifLearnZPInSpike)
        self.register_buffer("currentAuxiliary", scale * zero_point / self.T)

        self.sigma = sigma

    def forward(self, x) -> Tensor:
        r"""

        :param x: (T, sizeB, lenSeq, dimFeat)
        :type x:Tensor
        :return:
        :rtype:Tensor
        """
        self.v = torch.ones_like(x[0]) * self.initial_mem
        spike_pot = []
        for t in range(self.T):
            self.v = self.v.detach() + x[t]
            output = MultiLevelFunction.apply(
                self.v, self.scale, self.L, self.sigma)
            self.v = self.v - output.detach()
            spike_pot.append(output - self.currentAuxiliary)

        return torch.stack(spike_pot, dim=0)


class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


funcFloor = FloorLayer.apply


class QCFSWithPrefix(nn.Module):
    """
    == 用于 ANN 中的量化层 ==
    """

    def __init__(self,
                 args: Namespace,
                 numFeatures: int,
                 nameLogger: str,
                 maxFloatFeats: Tensor = None) -> None:
        r"""
        :param args:
        :type args: Namespace
        :param maxFloatFeats: TODO (sizeB, lenSeq) or (lenSeq, dimFeat)??
        :type maxFloatFeats: Tensor
        :param numFeatures: 如果是输入特征量化，则为对应原始 Linear 层的输入特征数，
                            如果为输出特征量化， 则为原始 Linear 层的输出特征数。
        :type numFeatures: int,
        :param nameLogger: 用于记录日志的 logger 的名字字符串。
        :type nameLogger: str
        """
        super().__init__()
        logger = logging.getLogger(nameLogger)
        # region 确定 scale 和 zero_point
        # TODO 这里来自于 Prefix 量化中为 input 计算的静态分支的 scale 和 zero_point，
        #  它那里的 scale 和 zero_point 都是可学习的。
        if args.ifMinMax:
            if maxFloatFeats is None:
                errorMessage = ValueError(
                    f"希望使用特征来计算 scale 和 zero_point 的时候，必须传入具体的 maxFloatFeats 值！")
                logger.error(errorMessage)
                raise errorMessage
            else:
                maxFeat = maxFloatFeats.amax(dim=-1, keepdim=True)
                self.scale = nn.Parameter(
                    (2 * maxFeat / (2 ** args.bitsForQuant - 1)).clamp(min=1e-4,
                                                                       max=1e4),
                    requires_grad=args.ifLearnScale)
                self.zero_point = nn.Parameter(
                    (2 ** args.bitsForQuant - 1) - 1 * torch.ones_like(self.scale),
                    requires_grad=args.ifLearnZP
                )
        else:
            self.scale = nn.Parameter(
                # torch.ones(size=()),
                torch.tensor(1 / (2 ** args.bitsForQuant - 1)),
                requires_grad=args.ifLearnScale
            )
            self.zero_point = nn.Parameter(
                torch.zeros(size=()),
                requires_grad=args.ifLearnZP
            )

        # endregion

        # region 用于前传中的超参
        self.shapeQuantized = [1, numFeatures]
        self.sizeGroup = numFeatures
        self.t: int = args.bitsForQuant  # 2^N -1
        # self.t = args.L * args.T
        self.phi: float = args.phi
        self.beta: float = args.betaForZP  # TODO 在argument 中设置，默认值 0.5，范围[0,1]
        self.gamma: float = args.gammaForZP  # TODO 在argument 中设置，默认值 0.5，范围[0,1]
        self.queryMin: float = 0
        self.queryMax: float = 2 ** (args.bitsForQuant - 1)
        # endregion

    def forward(self, x) -> Tensor:
        r"""
        :param x: (sizeBatch, lenSeq, dimFeat)
        :type x: Tensor
        :return: (sizeBatch, lenSeq, dimFeat)
        :rtype: Tensor
        """
        round_zero_point = clamp_ste(
            round_ste(self.zero_point),
            self.queryMin,
            self.queryMax)
        x = floor_ste(x / self.scale)
        x = x.add(round_zero_point)
        x = x.clamp(self.queryMin, self.queryMax)
        x = x.sub(round_zero_point)
        x = x.mul(self.scale)
        return x


# region 暂时废弃的算子
class MyFloor(nn.Module):
    def __init__(self, args: Namespace, up: float = 1.):
        r"""

        :param args:
        :param up: 输入特征中的可能的最大值与最小值的差。在我们的例子中，输入是 0 到 1 的均匀分布采样的值。所以是 1.
        """
        super().__init__()
        self.up = up
        self.t: int = args.bitsForQuant
        self.zero_point = 2 ** (args.bitsForQuant - 1)

    def forward(self,
                x: Tensor):
        x = x / self.up
        x = funcFloor(x * self.t) + self.zero_point
        x = torch.clamp(x, 0, self.t)
        x = x * self.up - self.up * self.zero_point
        return x


# class LMHTNeuron(nn.Module):
#     def __init__(self, L: int, T=2, th=1., initial_mem=0.):
#         super(LMHTNeuron, self).__init__()
#         self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=False)
#         self.v = None
#         self.initial_mem = initial_mem * th
#         self.L = L
#         self.T = T
#         self.scale = 1.
#
#     def forward(self, x):
#         self.v = torch.ones_like(x[0]) * self.initial_mem
#         x = x * self.scale
#         spike_pot = []
#         for t in range(self.T):
#             self.v = self.v.detach() + x[t]
#             output = MultiLevelFunction.apply(self.v, self.v_threshold, self.L)
#             self.v -= output.detach()
#             spike_pot.append(output)
#
#         return torch.stack(spike_pot, dim=0)


# class MultiLevelFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, th, L, sigma=0.5):
#         k = (input / th).floor().clamp(0, L)
#         out = k * th
#         ctx.save_for_backward(input, th)
#         ctx.L = L
#         ctx.sigma = sigma
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, th = ctx.saved_tensors
#         L = ctx.L
#         sigma = ctx.sigma
#
#         grad_input = torch.zeros_like(input)
#         for k in range(1, L + 1):
#             center = k * th
#         surrogate = torch.sigmoid((input - center) / sigma)
#         grad_surrogate = surrogate * (1 - surrogate) / sigma
#         grad_input += grad_surrogate
#
#         grad_input = grad_output * grad_input
#         return grad_input, None, None, None


# endregion


# region正负脉冲的算子 by Gt
class MultiLevelFunctionPN(Function):
    @staticmethod
    def forward(ctx, input, th, L):
        k = (input / th).clamp(-L, L)
        # 进行标记
        # 对正部分应用floor, 负部分为0
        k_pos = torch.where(k > 0, k.floor(), torch.tensor(0.0))
        # 对负部分应用ceil, 正部分为0
        k_neg = torch.where(k < 0, k.ceil(), torch.tensor(0.0))
        k = k_pos + k_neg  # 合并正负部分
        out = k * th
        # k = ((input / th).floor() + zero_point).clamp(0, L)

        mask = ((input.detach() >= (L + 1) * th) *
                (input.detach() <= (L + 1) * th)).float()
        ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None, None


class LMHTNeuronPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 scale: float = 1.):
        r"""

        :param L:
        :param T:
        :param th:
        :param initial_mem:
        :param sigma:
        :param scale: 这里的 scale 应该是前一层数据记录的 scale？
        """
        super().__init__()
        self.v_threshold = nn.Parameter(torch.tensor([args.Thresh]),
                                        requires_grad=True)
        self.v: Tensor
        self.initial_mem = args.initial_mem * args.Thresh
        self.L = args.L
        self.T = args.T
        self.scale = scale
        self.averageT = args.averageT
        self.act = MultiLevelFunctionPN.apply

    def forward(self, x):
        self.v = torch.zeros(x.shape[1:])
        spike_pot = list()
        if self.averageT:
            x_mean_along_T = x.mean(dim=0, keepdim=False)
        else:
            x_mean_along_T = None
        for t in range(self.T):
            if self.averageT:
                self.v += x_mean_along_T
            else:
                self.v += x[t]
            output = self.act(self.v,
                              self.v_threshold,
                              self.L,
                              self.sigma)
            self.v -= output
            spike_pot.append(output)
        return torch.stack(spike_pot, dim=0)


class LMHTNeuronWithTriSpike(nn.Module):
    def __init__(self, L, T=2, th=1.0, initial_mem=0.0, sigma=0.5, scale=1.0, pulse_pos=3, pulse_neg=-2):
        super(LMHTNeuronWithTriSpike, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=True)
        self.v = None
        self.initial_mem = initial_mem * th
        self.L = L
        self.T = T
        self.sigma = sigma
        self.scale = scale
        self.pulse_pos = pulse_pos
        self.pulse_neg = pulse_neg
        # 继续使用多级量化激活函数
        self.act = MultiLevelFunction.apply

    def forward(self, x):
        self.v = torch.ones_like(x[0]) * self.initial_mem
        spike_pot = []

        for t in range(self.T):
            self.v = self.v + x[t]
            output = self.act(self.v, self.v_threshold, self.L, self.sigma)

            # 发射三元脉冲：正脉冲、负脉冲或无脉冲
            output_spikes = torch.zeros_like(output)
            output_spikes[output >= self.v_threshold] = self.pulse_pos
            output_spikes[output <= -self.v_threshold] = self.pulse_neg

            # 重置膜电位
            self.v = self.v - output_spikes

            spike_pot.append(output_spikes)

        return torch.stack(spike_pot, dim=0)
# endregion
