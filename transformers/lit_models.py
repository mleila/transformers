'''torch lightning modules.'''
from argparse import ArgumentParser

import torch
from torch import nn
import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    """Simple PyTorch-Lightning model to train our Transformer."""

    def __init__(self, model, padding_index, learning_rate, batch_size):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'batch_size')
        self.model = model
        self.loss = nn.CrossEntropyLoss(ignore_index=padding_index)

    def training_step(self, batch, batch_ind):
        x, y = batch.src, batch.trg
        logits = self.model(x, y)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_ind):
        x, y = batch.src, batch.trg
        logits = self.model(x, y)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=float, default=0.0001)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
