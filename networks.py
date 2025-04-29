import logging
from argparse import Namespace
from typing import Sequence, Union

import torch
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.neuron import LIFNode
from torch import nn, Tensor

from layers import MLSWATNeuron


class TinySNN(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int],
                 scale: Tensor,
                 zero_point: Union[float, Tensor],
                 args: Namespace,
                 nameLogger: str):
        r"""

        :param hidden_sizes:
        :type hidden_sizes: Sequence[int]
        :param scale:
        :type scale:Tensor
        :param zero_point:
        :type zero_point:Union[float,Tensor]
        :param args:
        :type args: Namespace:
        :param nameLogger:
        :type nameLogger:str
        """
        super().__init__()
        self.logger = logging.getLogger(nameLogger)
        self.T = args.T
        # self.L: int = args.bitsForQuant // args.T
        self.L = args.L
        if args.neuron == "LIF":
            self.spike1 = LIFNode(detach_reset=True)

            self.spike2 = LIFNode(detach_reset=True)
        elif args.neuron == "LMHT":
            self.spike1 = MLSWATNeuron(args, scale=scale, zero_point=zero_point, nameLogger=nameLogger)
            self.spike2 = MLSWATNeuron(args, scale=scale, zero_point=zero_point, nameLogger=nameLogger)
            # self.spike3 = MLSWATNeuron(args, scale=scale, zero_point=zero_point, nameLogger=nameLogger)
        else:
            raise NotImplementedError(f"只支持 LIF 和 LMHT 神经元。")
        self.linear1 = layer.Linear(hidden_sizes[0], hidden_sizes[1], True, "m")
        self.linear2 = layer.Linear(hidden_sizes[1], hidden_sizes[2], True, "m")
        self.linear3 = layer.Linear(hidden_sizes[2], hidden_sizes[3], True, "m")
        # self.linear4 = layer.Linear(hidden_sizes[3], hidden_sizes[4], True, "m")
        self.to_spikes = MLSWATNeuron(args, scale=scale, zero_point=zero_point, nameLogger=nameLogger)

    def forward(self, x):
        r"""

        :param x: (B, lenSeq, dimSample)
        :return: (T, B, lenSeq, dimSample')
        """
        # 把 x 放到一个长度为 T 的列表里，再 stack
        x = torch.stack([x] * self.T, dim=0)
        # X_ = self.L * x.unsqueeze(0).repeat(self.T, 1, 1, 1)
        x = self.to_spikes(x)
        temp = x.mean(dim=0, keepdim=False)
        self.logger.debug(f"从第一层LMHT 出来的特征的信息：")
        self.logger.debug(f"形状：{x.shape}；先沿着时间轴取均值。")
        self.logger.debug(f"范围：[{temp.min().item():.6f}, {temp.max().item():.6f}]")
        self.logger.debug(f"均值：{temp.mean().item():.6f}, 标准差：{temp.std().item():.6f}")
        x = self.linear1(x)
        x = self.spike1(x)
        x = self.linear2(x)
        x = self.spike2(x)
        x = self.linear3(x)
        # x = self.spike3(x)
        # x = self.linear4(x)
        return x
