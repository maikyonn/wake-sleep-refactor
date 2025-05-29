import logging
import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Ensure these imports point to the new DataModule file if you rename it
from src.MidiDataModule import MidiDataModule, DataCfg 
from src.StariaTokenizer import MusicTokenizerWithStyle
from src.StariaModule import StariaModel # Assuming StariaModule is your PyTorch Lightning model

# Environment variable for better C++ stack traces if SIGABRT occurs
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = "1" 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Uncomment for detailed CUDA errors (slows down)


torch.set_float32_matmul_precision('medium')

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

class GPUMemoryCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available() and trainer.global_rank == 0:
            # Use trainer.strategy.local_rank for the correct device index in DDP
            device_index = trainer.strategy.local_rank if hasattr(trainer.strategy, 'local_rank') else 0
            used_gb = torch.cuda.memory_reserved(device_index) / 2**30
            pl_module.log('gpu_GB_reserved_rank0', used_gb, on_step=True, prog_bar=True, rank_zero_only=True)
            alloc_gb = torch.cuda.memory_allocated(device_index) / 2**30
            pl_module.log('gpu_GB_allocated_rank0', alloc_gb, on_step=True, prog_bar=False, rank_zero_only=True)

        if _HAS_PSUTIL and trainer.global_rank == 0:
            sys_mem = psutil.virtual_memory()
            sys_used_gb = (sys_mem.total - sys_mem.available) / 2**30
            pl_module.log('ram_GB_rank0', sys_used_gb, on_step=True, prog_bar=True, rank_zero_only=True)
        elif trainer.global_rank == 0 and batch_idx == 0 and not _HAS_PSUTIL:
            print("psutil not installed, system memory usage will not be logged.")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available() and trainer.global_rank == 0:
            device_index = trainer.strategy.local_rank if hasattr(trainer.strategy, 'local_rank') else 0
            used_gb = torch.cuda.memory_reserved(device_index) / 2**30
            pl_module.log('val_gpu_GB_reserved_rank0', used_gb, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
        if _HAS_PSUTIL and trainer.global_rank == 0:
            sys_mem = psutil.virtual_memory()
            sys_used_gb = (sys_mem.total - sys_mem.available) / 2**30
            pl_module.log('val_ram_GB_rank0', sys_used_gb, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, 
                        help="Path to the training data .pkl file (containing paths).")
    parser.add_argument('--val_data_dir', type=str, default=None, 
                        help="Path to the validation data .pkl file (containing paths, optional).")
    parser.add_argument('--max_len', type=int, default=4096, help="Max token sequence length after tokenization.")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2, help="Num workers per GPU for DataLoader.")
    parser.add_argument('--seq_limit', type=int, default=None, 
                        help="Limit total samples from pkl to use (for debugging map-style dataset).")
    
    # Model args (copied from your script)
    parser.add_argument('--encoder_dim', type=int, default=1536)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--encoder_heads', type=int, default=8)
    parser.add_argument('--decoder_dim', type=int, default=1536)
    parser.add_argument('--decoder_depth', type=int, default=16)
    parser.add_argument('--decoder_heads', type=int, default=24)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--accum_grads', type=int, default=4)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    # Hardware args
    parser.add_argument('--gpus', type=int, default=-1, help="Num GPUs: -1 for all, 0 for CPU.")
    parser.add_argument('--nodes', type=int, default=1)
    
    # Logging args
    parser.add_argument('--wandb_project', type=str, default='staria_model')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--disable_wandb', action='store_true')
    
    # Task/Mode for DataCfg (can be made arguments if they change often)
    parser.add_argument('--data_mode', type=str, default="synthetic", choices=["synthetic", "real"])
    parser.add_argument('--data_task', type=str, default="generative", choices=["generative", "contrastive", "classification"])
    parser.add_argument('--use_snippet', action='store_true', help="Enable snippet mode for generative task")


    args = parser.parse_args()

    if args.gpus == 0: accelerator, devices = 'cpu', 1
    elif torch.cuda.is_available(): accelerator, devices = 'gpu', args.gpus if args.gpus != -1 else torch.cuda.device_count()
    else: accelerator, devices = 'cpu', 1; print("CUDA not available, using CPU.")

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s %(process)d: %(message)s')
    pl.seed_everything(42, workers=True)
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"Effective devices: {devices} on accelerator: {accelerator}")
    main_logger.info(f"Starting training with resolved args: {args}")

    tokenizer = MusicTokenizerWithStyle()
    data_cfg = DataCfg(
        data_dir=args.data_dir,
        val_data_dir=args.val_data_dir,
        mode=args.data_mode,
        task=args.data_task,
        use_snippet=args.use_snippet,
        max_len=args.max_len,
        seq_limit=args.seq_limit,
        shuffle_records=True, # Shuffle records for training
        skip_long_after_tokenization=True,
        # AugmentCfg can be specified here if needed, e.g.:
        # augment=AugmentCfg(enable=True, pitch=3, velocity=5, tempo=0.05)
    )
    dm = MidiDataModule(
        cfg=data_cfg, tokenizer=tokenizer,
        batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=True # Usually True for training for consistent batch sizes
    )
    dm.setup("fit")

    model = StariaModel( # Ensure StariaModel is defined or imported correctly
        tokenizer=tokenizer, encoder_dim=args.encoder_dim, encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads, decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads, max_len=args.max_len, lr=args.lr
    )

    if args.disable_wandb:
        trainer_logger = pl.loggers.CSVLogger("logs", name="staria_mapstyle_training")
    else:
        trainer_logger = WandbLogger(project=args.wandb_project, name=args.wandb_name, log_model="all")

    callbacks_list = []
    checkpoint_dir = os.path.join('checkpoints', trainer_logger.name if trainer_logger and hasattr(trainer_logger, 'name') else "mapstyle_run")
    
    if args.val_data_dir:
        ckpt_cb = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3,
                                  filename='{epoch:02d}-{val_loss:.4f}', dirpath=checkpoint_dir)
        callbacks_list.append(ckpt_cb)
    else:
        main_logger.warning("No val_data_dir. Saving last checkpoint only for training.")
        callbacks_list.append(ModelCheckpoint(save_last=True, filename='last-{epoch:02d}', dirpath=checkpoint_dir))
    
    callbacks_list.append(GPUMemoryCallback())
    callbacks_list.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(
        logger=trainer_logger, callbacks=callbacks_list, max_epochs=args.epochs,
        accelerator=accelerator, devices=devices, num_nodes=args.nodes,
        strategy='ddp_find_unused_parameters_true', 
        precision='bf16-mixed',
        accumulate_grad_batches=args.accum_grads,
        val_check_interval=1.0 if args.val_data_dir else 0.0, # Check every epoch if val, else no val
        gradient_clip_val=1.0,
        # Consider enabling deterministic mode for deeper debugging if needed, but it can affect performance
        # deterministic=True, 
    )

    main_logger.info("Starting Pytorch Lightning Trainer.fit()...")
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_from_checkpoint)
    
    if trainer.global_rank == 0:
        final_ckpt_path = os.path.join(checkpoint_dir, "final_model.ckpt")
        trainer.save_checkpoint(final_ckpt_path)
        main_logger.info(f"Training finished. Final checkpoint: {final_ckpt_path}")

if __name__ == '__main__':
    main()