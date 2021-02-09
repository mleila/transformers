'''torch lightning data modules. Each module correspond to a dataset'''
import os
from argparse import ArgumentParser
from pydantic.errors import DataclassTypeError

import spacy
import pandas as pd

import torch
import torchtext
from torchtext.datasets import WMT14
import pytorch_lightning as pl

from transformers.nlp_utils import make_spacy_tokenizer, split_tokenizer


class WMT_DataModule(pl.LightningDataModule):
    """
    This Module encapsulates all logic needed to download, process, and batch
    the MWT14 dataset for Machine Translation Tasks
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        reduce: bool,
        tokenizer: str,
        max_sent_len: int,
        max_data_size: int,
        src_vocab_max_size: int,
        trgt_vocab_max_size: int
        ):

        super().__init__()

        # general parameters
        self.data_dir = data_dir
        self.batch_size = batch_size

        # data reduction parameters
        self.reduce = reduce
        self.max_sent_len = max_sent_len
        self.max_data_size = max_data_size
        self.src_vocab_max_size = src_vocab_max_size
        self.trgt_vocab_max_size = trgt_vocab_max_size

        # define constants
        self.reduced_train = 'reduced_train'
        self.reduced_valid = 'reduced_valid'

        # set selected tokenizers
        self._set_tokenizer(tokenizer)

    def _set_tokenizer(self, tokenizer):
        """
        Set tokenizers for English and German.
        Args:
            tokenizer: choice of toenizer ("spacy", "simple")
            the simple tokenizer just splits the data by whitespace
        """
        if tokenizer == 'spacy':
            self._english_tokenizer = make_spacy_tokenizer('en_core_web_sm')
            self._german_tokenizer = make_spacy_tokenizer('de_core_news_sm')
        elif tokenizer == 'simple':
            self._english_tokenizer = split_tokenizer
            self._german_tokenizer = split_tokenizer
        else:
            raise NameError(f'tokenizer {tokenizer} is not supported')

    def _reduce_data(self, source_file_name, target_file_name, max_data_size):
        '''
        reduce the size of the training set
        '''
        en_source = os.path.join(self.data_dir, 'wmt14', source_file_name + '.en')
        de_source = os.path.join(self.data_dir, 'wmt14', source_file_name + '.de')

        english_df = pd.read_csv(en_source, sep='\n', header=None)
        german_df = pd.read_csv(de_source, sep='\n', header=None)

        df = pd.concat([english_df, german_df], axis=1)
        df.columns = ['english', 'german']

        # restrict dataframe size
        if self.max_data_size:
            df = df.sample(n=max_data_size)

        # preprocessing : move out of this func
        df.english = df.english.str.lower()
        df.german = df.german.str.lower()

        # remove very long sentence
        df['english_sent_len'] = df.english.apply(self._english_tokenizer).agg(len)
        df['german_sent_len'] = df.german.apply(self._german_tokenizer).agg(len)
        df = (df
            .query(f'english_sent_len <= {self.max_sent_len} and german_sent_len <= {self.max_sent_len}')
            )

        en_target = os.path.join(self.data_dir, 'wmt14', target_file_name + '.en')
        de_target = os.path.join(self.data_dir, 'wmt14', target_file_name + '.de')

        df['english'].to_csv(en_target, sep='\n', header=None, index=False)
        df['german'].to_csv(de_target, sep='\n', header=None, index=False)

        print(f'number of sentence pairs in reduced dataset = {len(df)}')

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or
        that need to be done only from a single GPU in distributed settings.
        Example:
            - Download dataset
            - Tokenize
        """
        # download
        WMT14.download(self.data_dir)

        # limit
        if self.reduce:
            # reduce training
            self._reduce_data(
                source_file_name='train.tok.clean.bpe.32000',
                target_file_name=self.reduced_train,
                max_data_size=self.max_data_size
                )

            # reduce validation
            self._reduce_data(
                source_file_name='newstest2013.tok.bpe.32000',
                target_file_name=self.reduced_valid,
                max_data_size=None
                )

    def setup(self, stage=None):
        '''
        There are also data operations you might want to perform on every GPU. Use setup to do things like:
        Example:
            - count number of classes
            - build vocabulary
            - perform train/val/test splits
            - apply transforms (defined explicitly in your datamodule or assigned in init)
        '''
        eos_token = '<eos>'
        self.src_field = torchtext.data.Field(
            tokenize=self._english_tokenizer,
            eos_token=eos_token,
            batch_first=True,
            lower=True,
            )
        self.trgt_field = torchtext.data.Field(
            tokenize=self._german_tokenizer,
            eos_token=eos_token,
            batch_first=True,
            lower=True
            )

        root = str(self.data_dir)
        train_data = self.reduced_train if self.reduce else 'train.tok.clean.bpe.32000'
        valid_data = self.reduced_valid if self.reduce else 'newstest2013.tok.bpe.32000'

        self.train, self.valid, self.test = WMT14.splits(
            exts=('.en', '.de'),
            fields=(self.src_field, self.trgt_field),
            root=root,
            train=train_data,
            validation=valid_data
            )

        self.src_field.build_vocab(self.train, max_size=self.src_vocab_max_size)
        self.trgt_field.build_vocab(self.train, max_size=self.trgt_vocab_max_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test, batch_size=self.batch_size)


    @classmethod
    def from_argparse_args(cls, args):
        """
        Use this method to instantiate
        """
        return cls(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            reduce=args.reduce,
            tokenizer=args.tokenizer,
            max_sent_len=args.max_sent_len,
            max_data_size=args.max_data_size,
            src_vocab_max_size=args.src_vocab_max_size,
            trgt_vocab_max_size=args.trgt_vocab_max_size
            )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data-dir', type=str)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--reduce', type=bool, default=True)
        parser.add_argument('--tokenizer', type=str, default='simple')
        parser.add_argument('--max_sent_len', type=int, default=30)
        parser.add_argument('--max_data_size', type=int, default=100_000)
        parser.add_argument('--src_vocab_max_size', type=int, default=10_000)
        parser.add_argument('--trgt_vocab_max_size', type=int, default=10_000)
        return parser
