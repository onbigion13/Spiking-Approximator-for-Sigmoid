import os
import time
from pprint import pformat

import torch.optim
from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data import DataModule
from models import PLModule
from utils import create_logger, get_args, nnModelCheckpoint

parser = get_args()
args = parser.parse_args()
device = 0
nameLogger = "training"
my_bar = TQDMProgressBar(refresh_rate=20)
optims = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam
}
# checkpoint_callback = ModelCheckpoint(
#     save_last=True,
#     save_top_k=3,
#     monitor="accVal",
#     mode="max",
#     filename="{epoch:02d}_{accVal:.4f}_PLModel",
# )
if __name__ == "__main__":
    versionLog = (f"Bits-{args.bitsForQuant}_T-{args.T}_L-{args.L}_Epoch-{args.numEpochs}"
                  f"_optimW-{args.optimW}-lrW-{args.lrInitW}_optimTh-{args.optimTh}-lrTh-{args.lrInitTh}"
                  f"_IntervalW-{args.numEpochsForW}_IntervalTh-{args.numEpochsForTh}")
    timeLocal = time.strftime("%Y-%m-%d_%X", time.localtime())
    os.makedirs(os.path.join(f"logs", versionLog), exist_ok=True)
    loggerOutside = create_logger(dirLog=os.path.join(f"logs", versionLog),
                                  t=timeLocal,
                                  name=nameLogger)
    loggerOutside.info(f"\nargs used:\n{pformat(vars(args))}\n")
    loggerTF = TensorBoardLogger(f"logs",
                                 name="",
                                 version=versionLog)

    dataModule = DataModule(args=args,
                            nameLogger=nameLogger)

    model = PLModule(args=args,
                     optims=optims,
                     # hidden_sizes=[args.dimSample, 512, 512, args.dimSample],
                     quant=dataModule.quant,
                     hidden_sizes=[1, 8, 16, 1],
                     nameLogger=nameLogger)
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=3,
        monitor="lossVal",
        mode="min",
        filename="{epoch:02d}_{lossVal:.8f}_PLModel",
    )

    net_checkpoint_callback = nnModelCheckpoint(
        filename='{epoch:02d}-{lossVal:.8f}_nnModel',
        save_top_k=3,
        monitor='lossVal',
        mode='min'
    )
    trainer = Trainer(default_root_dir='./logs',
                      max_epochs=args.numEpochs,
                      devices=[device, ],
                      accelerator='gpu',
                      # strategy=DDPStrategy(find_unused_parameters=True),
                      # strategy='ddp',
                      logger=loggerTF,
                      callbacks=[my_bar, checkpoint_callback, net_checkpoint_callback],
                      log_every_n_steps=args.printEvery,
                      )
    # region 训练阶段
    loggerOutside.info(f"开始训练和验证。\n")
    trainer.fit(model=model, datamodule=dataModule)
    loggerOutside.info(f"训练结束。")
    # endregion
