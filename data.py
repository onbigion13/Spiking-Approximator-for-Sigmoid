import logging
from argparse import Namespace
from typing import Tuple, Union, List

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn, Tensor
from torch.utils.data import Dataset, random_split, DataLoader

from layers import QCFSWithPrefix

valid_funcs = ["Sigmoid"]


class DataModule(LightningDataModule):
    def __init__(self,
                 args,
                 nameLogger: str, ):
        super().__init__()
        self.logger = logging.getLogger(nameLogger)
        self.args = args
        datasetWhole = TinyDataset(args=args,
                                   nameLogger=nameLogger)
        self.quant = datasetWhole.quant
        self.datasets = func_split_datasets(datasetsWhole=datasetWhole,
                                            ratios=args.ratios,
                                            seed=args.seed)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.logger.info(f"为训练以及验证阶段准备DataLoader")
        elif stage == "validate":
            self.logger.info(f"为验证阶段准备DataLoader")
        elif stage == "test":
            self.logger.info(f"为测试阶段准备DataLoader")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.datasets[0],
                          batch_size=self.args.sizeBatch,
                          pin_memory=True,
                          num_workers=self.args.numWorkers,
                          shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.datasets[1],
                          batch_size=2 * self.args.sizeBatch,
                          pin_memory=True,
                          num_workers=self.args.numWorkers,
                          shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.datasets[2],
                          batch_size=2 * self.args.sizeBatch,
                          pin_memory=True,
                          num_workers=self.args.numWorkers,
                          shuffle=False)


class TinyDataset(Dataset):
    def __init__(self,
                 args: Namespace,
                 nameLogger: str):
        super().__init__()
        self.logger = logging.getLogger(nameLogger)
        self.numSamples = args.numSamples
        self.nBits = args.bitsForQuant
        # self.sizeSample = (args.lenSeq, args.dimSample)
        self.sizeSample = (args.dimSample,)
        self.funcStr = args.funcToBeLearned
        self.quant = QCFSWithPrefix(args=args,
                                    numFeatures=1,
                                    nameLogger=nameLogger,
                                    maxFloatFeats=torch.rand(size=())-0.5)
        self._prepare_samples()

    def __len__(self):
        return self.numSamples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        r"""

        :param idx:
        :return: sample and target: [lenSeq, dimSample]
        """
        sample = self.samples[idx]
        target = self.func(self.quant(sample))
        return sample, target

    def _prepare_samples(self) -> None:
        self.logger.info(f"共 {self.numSamples} 个随机样本，逼近的函数是： {self.funcStr}")
        # self.samples = torch.randint(low=0, high=2 ** self.nBits + 1, size=(self.numSamples, *self.sizeSample))
        self.samples = torch.rand(size=(self.numSamples, *self.sizeSample))
        if self.funcStr == "Sigmoid":
            self.func = nn.Sigmoid()
        elif self.funcStr in [valid_funcs]:
            pass
        else:
            errorMsg = NotImplementedError(f"输入的需要逼近的函数 ：{self.funcStr} 不受支持，"
                                           f" 请从 {valid_funcs} 中选择需要逼近的函数。")
            self.logger.error(errorMsg)
            raise errorMsg
        self.logger.info("数据集构造完成。")


def func_split_datasets(datasetsWhole: Dataset,
                        ratios=None,
                        seed: int = 42) -> Union[Tuple[Dataset, Dataset, Dataset], List[Dataset]]:
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset=datasetsWhole,
                        lengths=ratios,
                        generator=generator)
