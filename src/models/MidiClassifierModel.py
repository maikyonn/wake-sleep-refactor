import os
import datetime
import logging
from argparse import ArgumentParser

overall_logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Accuracy
from transformers import BertConfig, BertModel, get_cosine_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger

from tokenizer import MusicTokenizerWithStyle, IGNORE_LABEL_IDX
from MidiDataModule import MidiDataModule, DataCfg, AugmentCfg
from utils import filter_significant_styles, extract_style_change_timestamps, condense_style_sequence

from transformers import (
    LongformerConfig,
    LongformerModel,
    LongformerTokenizerFast,
    get_cosine_schedule_with_warmup
)

from x_transformers import TransformerWrapper, Encoder
class MidiClassifier(pl.LightningModule):
    def __init__(self,
        vocab_size: int,
        n_classes: int,
        lr: float,
        max_length: int,
        dropout_rate: float,
        pad_id: int,
        hidden: int = 1024,
        layers: int = 12,
        heads: int = 16,
        macaron: bool = False,
        attn_flash: bool = False,
        ff_glu: bool = False,
        pre_norm: bool = False,
        residual_attn: bool = True,
        layer_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
    ):
        super().__init__()
        # Save all model settings as hyperparameters for reproducibility
        self.save_hyperparameters()

        # Build TransformerWrapper with FlashAttention + Macaron-style FFNs
        self.transformer = TransformerWrapper(
            num_tokens    = vocab_size,
            max_seq_len   = max_length,
            emb_dropout   = dropout_rate,
            attn_layers   = Encoder(
                dim             = hidden,
                depth           = layers,
                heads           = heads,
                macaron         = macaron, 
                attn_flash      = attn_flash,  
                ff_glu          = ff_glu,
                layer_dropout   = layer_dropout,
                attn_dropout    = attn_dropout,
                ff_dropout      = ff_dropout,
                pre_norm        = pre_norm,
                residual_attn   = residual_attn,
            )
        )

        # Final token‐wise classifier
        # Classification head: vocab_size -> hidden -> hidden -> n_classes
        self.classifier_head = torch.nn.Sequential(
            torch.nn.Linear(vocab_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden, n_classes, bias=False)
        )

        self.train_acc  = Accuracy(task='multiclass', num_classes=n_classes,
                                   ignore_index=IGNORE_LABEL_IDX)
        self.val_acc    = Accuracy(task='multiclass', num_classes=n_classes,
                                   ignore_index=IGNORE_LABEL_IDX)

    def forward(self, x, mask=None):
        hidden = self.transformer(x, mask=mask)
        return self.classifier_head(hidden)

    def common_step(self, batch, stage):
        x, mask, labels = (batch['input_ids'],
                           batch['attention_mask'],
                           batch['style_label_indices'])
        mask = mask.bool()
        logits = self(x, mask)
        loss   = F.cross_entropy(logits.view(-1, self.hparams.n_classes),
                                 labels.view(-1),
                                 ignore_index=IGNORE_LABEL_IDX)
        acc = self.train_acc if stage=='train' else self.val_acc
        acc(logits.view(-1, self.hparams.n_classes), labels.view(-1))
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f'{stage}_acc',  acc,   prog_bar=True, on_epoch=True, sync_dist=True)
        if stage=='train':
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'],
                     on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, idx):    return self.common_step(batch, 'train')
    def validation_step(self, batch, idx):  return self.common_step(batch, 'val')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # steps  = self.trainer.estimated_stepping_batches
        # warmup = int(0.1 * steps)
        # sched  = get_cosine_schedule_with_warmup(opt, warmup, steps)
        # return {
        #     'optimizer': opt,
        #     'lr_scheduler': {'scheduler': sched, 'interval': 'step'}
        # }
        return opt