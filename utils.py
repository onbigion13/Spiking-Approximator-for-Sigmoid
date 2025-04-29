import argparse
import logging
import os
import sys

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from termcolor import colored
from torch import Tensor
from torch.optim import Optimizer


def create_logger(dirLog: str,
                  t: str,
                  dist_rank: int = 0,
                  name: str = "training"):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + \
                ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers

    file_handler = logging.FileHandler(os.path.join(dirLog, f'{name}_{t}.log'),
                                       mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


class nnModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer: Trainer, filepath):
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        if trainer.is_global_zero:
            torch.save(model_to_save.net.state_dict(),
                       filepath)
        model_to_save.loggerMine.info(f"得到更佳的nn模型，保存至：\n{filepath}\n")
        trainer.strategy.barrier("CustomModelCheckpoint.save_checkpoint")


def load_yaml_config(yaml_file: str):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def get_args():
    parser = argparse.ArgumentParser(f"Approximate Sigmoid with tiny SNN.")
    # parser.add_argument('--config', type=str,
    #                     default="/home/duxiao/Workspace/Spiking-DeepSeek/configs/llama-2-7b-w4a4-t2-200.yaml",
    #                     help='YAML config file path')

    # region 数据相关参数。

    parser.add_argument("--dimSample", type=int, default=1,
                        help="")
    parser.add_argument("--lenSeq", type=int, default=1024,
                        help="")
    parser.add_argument("--numSamples", type=int, default=10000,
                        help="")
    parser.add_argument("--sizeBatch", type=int, default=32,
                        help="batch size.")
    parser.add_argument("--funcToBeLearned", type=str, default="Sigmoid")
    parser.add_argument("--ratios", type=int, nargs='+',
                        default=[0.8, 0.1, 0.1])
    parser.add_argument("--numWorkers", type=int, default=8)
    # endregion

    # region 训练相关参数。
    parser.add_argument("--trainStrategy", type=float, default=None)
    parser.add_argument("--numEpochs", type=int, default=100)
    parser.add_argument("--printEvery", type=int, default=2)

    # endregion

    # region 量化相关的参数
    parser.add_argument("--betaForZP", type=float, default=0.5,
                        help="范围 [0,1]")
    parser.add_argument("--gammaForZP", type=float, default=0.5,
                        help="范围 [0,1]")
    parser.add_argument("--bitsForQuant", type=int, default=3)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--ifMinMax", type=bool, default=False,
                        help="是否使用 maxFloatFeats 来计算张量的 scale 和 zero_point, 否则后二者都是标量。")
    parser.add_argument("--ifLearnScale", type=bool, default=False,
                        help="是否学习 scale")
    parser.add_argument("--ifLearnZP", type=bool, default=False,
                        help="是否学习 zero_point")

    # endregion

    # region 脉冲神经元相关的参数。
    parser.add_argument("--thresh", type=float, default=1 / 8)
    parser.add_argument("--threshes", type=float, nargs='+', default=[1 / 8 for _ in range(8)])
    parser.add_argument("--T", type=float, default=8)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--neuron", type=str, default="LMHT")
    parser.add_argument("--initial_mem", type=float, default=0.)
    parser.add_argument("--averageT", type=bool, default=False)
    parser.add_argument("--ifLearnScaleInSpike", type=bool, default=True,
                        help="LMHT 中是否学习 scale")
    parser.add_argument("--ifLearnZPInSpike", type=bool, default=True,
                        help="LMHT 中是否学习 zero_point")
    parser.add_argument("--ifLearnThreshInMLSWA", type=bool, default=True,
                        help="MLSWA 中是否学习 scale")
    parser.add_argument("--ifLearnZPinMSWATInMLSWA", type=bool, default=True,
                        help="MLSWA 中是否学习 zero_point")

    # endregion

    # region 优化相关参数
    parser.add_argument("--optimW", type=str, default="SGD")
    parser.add_argument("--optimTh", type=str, default="Adam")
    parser.add_argument("--lrInitW", type=float, default=0.1)
    parser.add_argument("--lrInitTh", type=float, default=0.0001)
    parser.add_argument("--numEpochsForW", type=int, default=7)
    parser.add_argument("--numEpochsForTh", type=int, default=3)

    parser.add_argument("--weight_decay", type=float, default=2e-5)
    # endregion

    # region 其他参数
    parser.add_argument("--seed", type=int, default=42)

    # endregion

    return parser


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    :param x:
    :type x: Tensor
    :return:
    :rtype:
    """
    return (x.round() - x).detach() + x


def floor_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    :param x:
    :type x: Tensor
    :return:
    :rtype:
    """
    return (x.floor() - x).detach() + x


def clamp_ste(x: torch.Tensor,
              minX: float,
              maxX: float) -> Tensor:
    r"""
    == 只让 x 的整数部分参与求导 ==

    :param x:
    :type x: Tensor
    :param minX:
    :type minX: float
    :param maxX:
    :type maxX :float
    :return:
    :rtype: Tensor
    """
    return (x.clamp(minX, maxX) - x).detach() + x


def get_grad_info(optimizer: Optimizer,
                  nameLogger: str):
    r"""

    :param optimizer:
    :type optimizer: Optimizer
    :param nameLogger:
    :type nameLogger: str:
    :return:
    """
    logger = logging.getLogger(nameLogger)
    total_grad = 0.0
    total_abs_grad = 0.0
    count = 0

    for group in optimizer.param_groups:
        logger.info(group["params"])
        for param in group['params']:
            if param.grad is not None:
                total_grad += param.grad.sum().item()
                total_abs_grad += param.grad.abs().sum().item()
                count += param.grad.numel()

    if count > 0:
        mean_grad = total_grad / count
        mean_abs_grad = total_abs_grad / count
        logger.info(f"Grad Mean: {mean_grad:.6f}, Grad Abs Mean: {mean_abs_grad:.6f}")
    else:
        logger.info("No gradients available.")
