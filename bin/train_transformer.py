#!/bin/env/python
'''
python bin/train_transformer.py \
    --data-dir datat_store \
    --batch-size 128 \
    --max-epochs 5 \
    --gpus 2 \
    --accelerator ddp
'''
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from transformers.lit_models import LitModel
from transformers.torch_models import Transformer
from transformers.lit_data import WMT_DataModule
from transformers.nlp_utils import split_tokenizer


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitModel.add_model_specific_args(parser)
    parser = WMT_DataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = WMT_DataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()

    # ------------
    # model
    # ------------
    target_vocab_size = len(dm.trgt_field.vocab)
    backbone = Transformer(num_classes=target_vocab_size, max_output_length=32)
    padding_index = dm.src_field.vocab.stoi[dm.src_field.pad_token]
    model = LitModel(backbone, padding_index, args.learning_rate, args.batch_size)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    #result = trainer.test(test_dataloaders=test_loader)
    #print(result)
    print('do some testing')


if __name__ == '__main__':
    cli_main()
