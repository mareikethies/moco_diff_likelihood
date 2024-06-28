import wandb
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from autofocus_model import AutofocusModel
from pytorch_lightning.loggers import WandbLogger
from autofocus_data_loader import MotionCTDataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


parser = ArgumentParser()

# add model specific args
parser = AutofocusModel.add_model_specific_args(parser)

# add data specific args
MotionCTDataLoader.add_data_specific_args(parser)

# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

# configure checkpointing to save both the best model wrt. val loss and the last model
checkpoint_callback_best_val = ModelCheckpoint(monitor='val/loss', save_last=True)
# log learning rate
lr_callback = LearningRateMonitor(logging_interval='epoch')
# wandb logger
wandb_logger = WandbLogger(project='ReferenceDiffusionLikelihood', name=args.run_name, log_model='all',
                           save_dir='./logs', settings=wandb.Settings(start_method='fork'))

trainer = Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback_best_val, lr_callback])

data = MotionCTDataLoader(**vars(args))
model = AutofocusModel(**vars(args))

trainer.fit(model, data)
trainer.test(ckpt_path='best', datamodule=data)
