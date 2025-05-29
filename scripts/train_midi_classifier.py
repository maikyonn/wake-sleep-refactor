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
from MidiClassifierModel import MidiClassifier

torch.set_float32_matmul_precision('medium')
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    pl.seed_everything(seed)


class GPUMemoryCallback(Callback):
    """Log GPU memory usage after each batch."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_reserved() / 2**30
            pl_module.log('gpu_memory_train_GB', used_gb, on_step=True, prog_bar=True)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_reserved() / 2**30
            pl_module.log('gpu_memory_val_GB', used_gb, on_step=False, on_epoch=True, prog_bar=True)


def main():
    parser = ArgumentParser()
    # Data args
    parser.add_argument('--data_dir', type=str, default='datasets/synth_midi_90k')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--limit_data', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=128)
    # Model args
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    # Hyperparameters for MidiClassifier
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--macaron', action='store_true', help='Use macaron style transformer blocks')
    parser.add_argument('--attn_flash', action='store_true', help='Use flash attention')
    parser.add_argument('--ff_glu', action='store_true', help='Use GLU in feedforward')
    parser.add_argument('--layer_dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--ff_dropout', type=float, default=0.1)
    # Training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--accum_grads', type=int, default=4)
    # Hardware args
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--nodes', type=int, default=1)
    # Logging args
    parser.add_argument('--wandb_project', type=str, default='midi_style')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default=None)

    args = parser.parse_args()

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    )

    # reproducibility
    set_seed(42)

    # DataModule
    data_cfg = DataCfg(
        data_dir=args.data_dir,
        mode='synthetic',
        max_len=args.max_length,
        seq_limit=args.limit_data,
        cache_dir=args.cache_dir,
        shuffle=True,
        skip_long=True,
        num_workers=args.num_workers,
        aug=AugmentCfg(enable=True, pitch=5, velocity=10, tempo=0.2, mixup=False),
    )
    dm = MidiDataModule(
        cfg=data_cfg,
        tokenizer=MusicTokenizerWithStyle(),
        batch_size=args.batch_size,
        val_split=0.1,
        seed=42,
        drop_last=True,
    )

    # Model
    model = MidiClassifier(
        vocab_size=MusicTokenizerWithStyle().vocab_size,
        n_classes=args.n_classes,
        lr=args.lr,
        max_length=args.max_length,
        dropout_rate=args.dropout_rate,
        pad_id=MusicTokenizerWithStyle().pad_id,
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        macaron=args.macaron,
        attn_flash=args.attn_flash,
        ff_glu=args.ff_glu,
        layer_dropout=args.layer_dropout,
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout,
    )

    # Logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        log_model=False,
    )
    
    # Create checkpoint directory based on wandb run name
    run_name = wandb_logger.name
    checkpoint_dir = os.path.join('checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=3, save_last=False,
        filename='epoch{epoch:02d}-val{val_loss:.2f}', dirpath=checkpoint_dir,
    )
    es_cb = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    gpu_cb = GPUMemoryCallback()

    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[ckpt_cb, es_cb, gpu_cb],
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy='ddp_find_unused_parameters_true' if args.gpus > 1 else None,
        precision='bf16-mixed',
        accumulate_grad_batches=args.accum_grads,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    # Fit
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt)

    # Finish
    overall_logger.info('Training complete')


if __name__ == '__main__':
    main()