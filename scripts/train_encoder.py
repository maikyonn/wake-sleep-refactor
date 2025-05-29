import os
import logging
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger

from src.StariaTokenizer import MusicTokenizerWithStyle, IGNORE_LABEL_IDX
from src.MidiDataModule import MidiDataModule, DataCfg, AugmentCfg
from src.StariaModule import ContrastiveEncoderLM

# ---------------------------------------------------------------------------
# Utility Callbacks
# ---------------------------------------------------------------------------
class GPUMemoryCallback(Callback):
    """Log GPU memory usage after each batch."""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_reserved() / 2**30
            pl_module.log('gpu_memory_train_GB', used_gb, on_step=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_reserved() / 2**30
            pl_module.log('gpu_memory_val_GB', used_gb, on_step=False, on_epoch=True, prog_bar=True)
# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    # data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.1)
    # model
    parser.add_argument('--dim', type=int, default=1536)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--temp', type=float, default=0.08)
    parser.add_argument('--lambda_local', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    # training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--accum_grads', type=int, default=4)
    # hardware
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--nodes', type=int, default=1)
    # logging
    parser.add_argument('--wandb_project', type=str, default='midi_contrastive')
    parser.add_argument('--wandb_name', type=str, default=None)
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    pl.seed_everything(42)

    # DataModule
    data_cfg = DataCfg(
        data_dir=args.data_dir,
        mode='synthetic',
        task='contrastive',
        max_len=args.max_len,
        val_split=args.val_split,
    )
    dm = MidiDataModule(cfg=data_cfg,
                        tokenizer=MusicTokenizerWithStyle(),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        drop_last=True)

    # Model
    enc_model = ContrastiveEncoderLM(
        tokenizer=MusicTokenizerWithStyle(),
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        max_len=args.max_len,
        temp=args.temp,
        lambda_local=args.lambda_local,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
    )

    # Logger
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_name)

    # checkpoints & callbacks
    ckpt_dir = os.path.join('checkpoints', wandb_logger.name or 'run')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=3,
                              filename='epoch{epoch:02d}-val{val/loss:.4f}', dirpath=ckpt_dir)
    es_cb = EarlyStopping(monitor='val/loss', patience=3, mode='min')
    gpu_cb = GPUMemoryCallback()

    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[ckpt_cb, es_cb, gpu_cb],
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy='ddp_find_unused_parameters_true' if args.gpus>1 else None,
        precision='bf16-mixed',
        accumulate_grad_batches=args.accum_grads,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
    )

    # Fit
    trainer.fit(enc_model, datamodule=dm)

if __name__ == '__main__':
    main()
