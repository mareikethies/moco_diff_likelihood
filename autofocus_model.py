import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning import LightningModule


class AutofocusModel(LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        print('HPARAMS: \n')
        print(self.hparams)

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Model')
        parser.add_argument('--run_name', type=str, default=None)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parent_parser

    def forward(self, image):
        out = self.model(image)

        return out

    def normalize_image(self, x):
        return torch.clamp((x - torch.min(x)) / (torch.max(x) - torch.min(x)), min=0.0, max=1.0)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        image, metric = batch
        if batch_idx == 0:
            wandb_image = wandb.Image(self.normalize_image(torch.squeeze(image[0, ...])), caption=f'Metric: {metric[0, ...]}')
            wandb.log({'training_examples': wandb_image})

        out = self.forward(image)

        loss = self.l1_loss(torch.squeeze(out), metric)

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        image, rpe = batch

        out = self.forward(image)

        loss = self.l1_loss(torch.squeeze(out), rpe)

        self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        self.model.eval()
        image, rpe = batch

        out = self.forward(image)

        loss = self.l1_loss(torch.squeeze(out), rpe)

        self.log('test/loss', loss)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams['learning_rate'])
        return optimizer
