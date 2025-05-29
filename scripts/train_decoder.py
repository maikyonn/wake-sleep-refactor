import logging
import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.StariaTokenizer import MusicTokenizerWithStyle, IGNORE_LABEL_IDX
from src.MidiDataModule import MidiDataModule, DataCfg, AugmentCfg
from src.StariaModule import DecoderLM

torch.set_float32_matmul_precision('medium')
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
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seq_limit', type=int, default=None)
    # model
    parser.add_argument('--dim', type=int, default=1536)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--heads', type=int, default=24)
    parser.add_argument('--cross_attend', action='store_true', help='Enable cross attention in decoder')
    parser.add_argument('--lr', type=float, default=3e-4)
    # training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--accum_grads', type=int, default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    # hardware
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--nodes', type=int, default=1)
    # logging
    parser.add_argument('--wandb_project', type=str, default='midi_decoder')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    pl.seed_everything(42)

    # DataModule
    data_cfg = DataCfg(
        data_dir=args.data_dir,
        mode='real',
        task='generative',
        max_len=args.max_len,
        val_split=args.val_split,
        seq_limit=args.seq_limit,
    )
    dm = MidiDataModule(cfg=data_cfg,
                        tokenizer=MusicTokenizerWithStyle(),
                        batch_size=args.batch_size,
                        num_workers=int(args.num_workers),
                        drop_last=True)

    # Model
    decoder_model = DecoderLM(
        tokenizer=MusicTokenizerWithStyle(),
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        max_len=args.max_len,
        cross_attend=args.cross_attend,
        lr=args.lr
    )
    

    # Logger
    if args.disable_wandb:
        logger = pl.loggers.CSVLogger("logs", name="decoder_training")
    else:
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_name)

    # checkpoints & callbacks
    run_name = "run" if args.disable_wandb else (logger.name or "run")
    ckpt_dir = os.path.join('checkpoints', run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=1,
                              filename='{epoch:02d}-{train_loss:.4f}', dirpath=ckpt_dir)
    # es_cb = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    gpu_cb = GPUMemoryCallback()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_cb, gpu_cb, lr_monitor],
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy='ddp' if args.gpus>1 else None,
        precision='bf16-mixed',
        accumulate_grad_batches=args.accum_grads,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
    )

    # Fit
    trainer.fit(decoder_model, datamodule=dm, ckpt_path=args.resume_from_checkpoint)

if __name__ == '__main__':
    main()
