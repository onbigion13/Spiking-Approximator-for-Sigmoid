import logging
from argparse import Namespace
from typing import Optional, Tuple, Sequence, Dict, Union

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from spikingjelly.activation_based import functional, layer
from torch import nn
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

__all__ = [
    "PLModule"
]

from layers import QCFSWithPrefix, MLSWATNeuron
from networks import TinySNN


class PLModule(LightningModule):
    def __init__(self,
                 args: Namespace,
                 optims: Dict[str, Union[type(SGD), type(Adam)]],
                 hidden_sizes: Sequence[int],
                 quant: QCFSWithPrefix,
                 nameLogger: str,
                 ) -> None:
        r"""

        :param args:
        :type args:Namespace:
        :param hidden_sizes:
        :type hidden_sizes: Sequence[int]
        :param quant:
        :type quant:nn.Module
        :param nameLogger:
        """
        super(PLModule, self).__init__()
        # 定义日志记录器
        self.AccTest = None
        self.AccTrain = None
        self.args = args
        self.optims = optims
        self.nameLogger = nameLogger
        self.loggerMine = logging.getLogger(nameLogger)
        self.funcLoss = nn.MSELoss()
        self.net = TinySNN(hidden_sizes=hidden_sizes,
                           args=args,
                           nameLogger=nameLogger,
                           scale=quant.scale.data,
                           zero_point=quant.zero_point.squeeze())
        # 保存超参。
        self.save_hyperparameters()
        self.automatic_optimization = False

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # region ---交替迭代优化的优化器定义---
        paramsW = []
        paramsTh = []
        for module in self.net.modules():  # 不用 named_modules，也可用 modules()
            if isinstance(module, layer.Linear):
                paramsW.extend(module.parameters())
            elif isinstance(module, MLSWATNeuron):
                paramsTh.extend(module.parameters())
        optimizerW = self.optims[self.args.optimW](lr=self.args.lrInitW, params=paramsW,
                                                   weight_decay=self.args.weight_decay)
        optimizerTh = self.optims[self.args.optimW](lr=self.args.lrInitTh, params=paramsTh,
                                                    weight_decay=self.args.weight_decay)
        schedulerW = CosineAnnealingLR(optimizerW, self.trainer.max_epochs)
        schedulerTh = CosineAnnealingLR(optimizerTh, self.trainer.max_epochs)
        self.loggerMine.info(f"网络权重使用的优化器为：\n{optimizerW}\n使用的学习率调度器为：\n{schedulerW}")
        self.loggerMine.info(f"网络权重使用的优化器为：\n{optimizerTh}\n使用的学习率调度器为：\n{schedulerTh}")
        return (
            [optimizerW, optimizerTh],
            [schedulerW, schedulerTh]
        )
        # endregion

        # region ---一起优化的优化器定义---
        # optimizer = SGD(params=self.net.parameters(),
        #                 lr=self.args.lrInit,
        #                 weight_decay=self.args.weight_decay)
        # lr_scheduler_config = {
        #     # # ReduceLROnPlateau
        #     # "scheduler": self.argsOpt['typeSch'](optimizer,
        #     #                                       mode=self.argsOpt['mode'],
        #     #                                       factor=self.argsOpt['factor'],
        #     #                                       patience=self.argsOpt['patience']),
        #
        #     # MultiStepLR
        #     # "scheduler": self.argsOpt['typeSch'](optimizer,
        #     #                                       milestones=self.argsOpt['milestones'],
        #     #                                       gamma=self.argsOpt['gamma']),
        #
        #     # StepLR
        #     # "scheduler": self.argsSch["typeSch"](optimizer,
        #     #                                      step_size=self.argsSch['step_size'],  # 64
        #     #                                      gamma=self.argsSch['gamma']),  # 0.1
        #
        #     # CosLR
        # "scheduler": CosineAnnealingLR(optimizer,
        #                                T_max=self.args.numEpochs,
        #                                eta_min=0),
        #
        #     "interval": "epoch",
        #     "frequency": 1,
        #     'monitor': 'lossVal'
        # }
        # endregion
        # return {"optimizer": optimizer,
        #         "lr_scheduler": lr_scheduler_config}

    def forward(self, x):
        r"""

        :param x:
        :return:
        """
        functional.reset_net(self.net)
        return self.net(x)

    # region 训练
    # def on_train_start(self) -> None:
    # self.AccTrain = Accuracy(task="multiclass")

    def on_train_epoch_start(self) -> None:
        self.loggerMine.info(f" ")
        if self.current_epoch % (self.args.numEpochsForTh + self.args.numEpochsForW) < self.args.numEpochsForW:
            self.loggerMine.info(f"第 {self.current_epoch} 个 epoch 优化权重。\n")
        else:
            self.loggerMine.info(f"第 {self.current_epoch} 个 epoch 优化权重。\n")

    # self.AccTrain.to(self.device)
    # self.AccTrain.reset()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        outputBatch, lossBatch, sizeBatch, labelsBatch = self.universal_step(batch, "train")

        # region ---交替迭代梯度下降和反传更新---
        optimizerW: Optimizer
        optimizerTh: Optimizer
        optimizerW, optimizerTh = self.optimizers()
        # 如果当前epoch在args.numEpochsForW，则优化Weight
        if self.current_epoch % (self.args.numEpochsForTh + self.args.numEpochsForW) < self.args.numEpochsForW:
            # 先优化 W numEpochsForW 个epoch
            optimizerW.zero_grad()
            self.manual_backward(lossBatch)
            optimizerW.step()
        else:
            # 然后优化 Th numEpochsForth 个epoch
            optimizerTh.zero_grad()
            self.manual_backward(lossBatch)
            optimizerTh.step()

        # endregion

        # accBatch = self.AccTrain(outputBatch, labelsBatch)

        self.log('lossTrain', lossBatch, prog_bar=True, batch_size=sizeBatch,
                 on_step=True, on_epoch=True, sync_dist=True)
        # self.log('accBatchTrain', accBatch, prog_bar=True, sync_dist=True, on_epoch=False, on_step=True,
        #          batch_size=sizeBatch)

        # if (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
        #     self.loggerMine.info(
        #         f"第{self.current_epoch}个epoch的第{batch_idx + 1}个训练批次的损失为：{lossBatch:.6f}")
        # self.loggerMine.info(
        # f"第{self.current_epoch}个epoch的第{batch_idx + 1}个训练批次的准确率为：{accBatch:.6f}\n")

        return {
            'loss': lossBatch,
            # 'acc': accBatch
        }

    def on_after_backward(self) -> None:
        pass
        # if self.current_epoch == 5:
        #     sys.exit(0)
        # optimizer = self.optimizers().optimizer
        # get_grad_info(optimizer, self.nameLogger)

    def on_train_epoch_end(self, ) -> None:
        #     pass

        # region ---交替迭代的学习率衰减---
        lrSchedulers = self.lr_schedulers()
        schedulerW, schedulerTh = lrSchedulers
        if self.current_epoch % (self.args.numEpochsForTh + self.args.numEpochsForW) < self.args.numEpochsForW:
            schedulerW.step()
        else:
            schedulerTh.step()
        avg_loss = self.trainer.callback_metrics["lossTrain_epoch"]
        self.loggerMine.info(f"第{self.current_epoch}个epoch的平均训练损失为：{avg_loss:.8f}\n\n")
        # endregion

        # accTrain = self.AccTrain.compute()
        # self.log('accTrain', accTrain, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        # self.loggerMine.info(f"第{self.current_epoch}个epoch的整体训练准确率为：{accTrain * 100:.2f}%")

    # endregion

    # region 验证
    # def on_validation_start(self) -> None:
    #     self.AccVal = Accuracy(task="multiclass")

    # def on_validation_epoch_start(self) -> None:
    #     self.AccVal.reset()
    #     self.AccVal.to(self.device)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        # 这里的 output 已经沿着T取均值了。
        outputBatch, lossBatch, sizeBatch, labelsBatch = self.universal_step(batch,
                                                                             "val")
        if (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
            self.loggerMine.info(f"第 {self.current_epoch} 个 epoch 的第 {batch_idx} 验证批次的信息如下：")
            self.loggerMine.info(f"Sigmoid 输出的形状: {labelsBatch.shape}")
            self.loggerMine.info(f"SNN 输出的形状: {outputBatch.shape}")

            # self.loggerMine.info(f"Sigmoid 输出的局部值：\n{labelsBatch[20:25, 500:505, 30:35]}")
            # self.loggerMine.info(f"SNN 输出的局部值: \n{outputBatch[20:25, 500:505, 30:35]}")
            # self.loggerMine.info(f"局部值的差异 label - output : "
            #                      f"\n{labelsBatch[20:25, 500:505, 30:35] - outputBatch[20:25, 500:505, 30:35]}")
            self.loggerMine.info(f"Sigmoid 输出的局部值：\n{labelsBatch[20:30]}")
            self.loggerMine.info(f"SNN 输出的局部值: \n{outputBatch[20:30]}\n")
            self.loggerMine.info(f"局部值的差异 label - output : \n{labelsBatch[20:30] - outputBatch[20:30]}\n")

        # accBatch = self.AccVal(outputBatch, labelsBatch).item()
        self.log(f'lossVal', lossBatch.item(), prog_bar=False, batch_size=sizeBatch,
                 on_step=True, on_epoch=True, sync_dist=True)
        # self.log(f'accValBatch', accBatch, prog_bar=False, batch_size=sizeBatch, sync_dist=True)

        return {
            'loss': lossBatch,
            # 'acc': accBatch
        }

    def on_validation_epoch_end(self) -> None:
        # accVal = self.AccVal.compute().item()
        # accVal = round(accVal, 6)

        # self.loggerMine.info(f"第{self.current_epoch}个epoch的整体验证准确率：{accVal}")

        # self.log('accVal', accVal, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        avg_loss = self.trainer.callback_metrics["lossVal_epoch"]
        self.loggerMine.info(f" ")
        self.loggerMine.info(f"第{self.current_epoch}个epoch的平均验证损失为：{avg_loss:.8f}")

    # endregion

    # region 测试
    # def on_test_start(self) -> None:
    # self.AccTest = Accuracy(task="multiclass")

    # def on_test_epoch_start(self) -> None:
    # self.AccTest.reset()
    # self.AccTest.to(self.device)

    # def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
    #     outputBatch, lossBatch, sizeBatch, labelsBatch = self.universal_step(batch,
    #                                                                          "test")
    # accBatch = self.AccTest(outputBatch, labelsBatch).item()
    # self.log(f'accTest', accBatch, prog_bar=False, batch_size=sizeBatch, sync_dist=True)
    # return {
    #     'acc': accBatch
    # }

    # def on_test_epoch_end(self) -> None:
    # accTest = self.AccTest.compute().item()
    # accTest.append(round(accTest, 6))
    # self.loggerMine.info(f"测试准确率：{accTest}")

    # endregion

    # region 工具函数
    def universal_step(self, batch, stage: str) -> Tuple:
        # data:(T,B,C,H,W), label:(B,)
        data, label = batch
        sizeBatch = label.size(0)

        output = self(data)
        outputMean = output.mean(0, keepdim=False)
        loss = self.funcLoss(outputMean, label)
        return outputMean, loss, sizeBatch, label

    # endregion
