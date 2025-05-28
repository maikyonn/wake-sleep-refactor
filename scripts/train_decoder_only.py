#!/usr/bin/env python
"""
Training Script for Decoder-Only Baseline

This script trains a pure decoder-only transformer on MIDI data with
snippet prompts. The model learns to generate structured music by
conditioning on snippet prompts that are prepended to training sequences.

Usage:
    python scripts/train_decoder_only.py --data_dir datasets/midi --epochs 50

Example sequence format:
<PROMPT_START> <A_SECTION> [snippet_A] <B_SECTION> [snippet_B] <PROMPT_END> [full_midi_sequence]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Import from new structure
import sys
import os
sys.path.append('.')
from staria.models.tokenizer import MusicTokenizerWithStyle
from staria.baselines.decoder_only import DecoderOnlyBaseline, PromptedMIDIDataset


class PromptedMIDIDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for prompted MIDI sequences.
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: MusicTokenizerWithStyle,
        batch_size: int = 4,
        num_workers: int = 4,
        max_len: int = 4096,
        snippet_length: int = 256,
        num_snippets: int = 2,
        train_val_split: float = 0.9,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.snippet_length = snippet_length
        self.num_snippets = num_snippets
        self.train_val_split = train_val_split
        
    def setup(self, stage: str = None):
        """Setup datasets."""
        # Get all MIDI files
        midi_files = []
        for ext in ['*.mid', '*.midi']:
            midi_files.extend(list(Path(self.data_dir).rglob(ext)))
        
        midi_files = [str(f) for f in midi_files]
        
        if not midi_files:
            raise ValueError(f"No MIDI files found in {self.data_dir}")
        
        # Split into train/val
        n_train = int(len(midi_files) * self.train_val_split)
        train_files = midi_files[:n_train]
        val_files = midi_files[n_train:]
        
        # Create datasets
        self.train_dataset = PromptedMIDIDataset(
            train_files,
            self.tokenizer,
            self.max_len,
            self.snippet_length,
            self.num_snippets
        )
        
        self.val_dataset = PromptedMIDIDataset(
            val_files,
            self.tokenizer,
            self.max_len,
            self.snippet_length,
            self.num_snippets
        )
        
        print(f"Created datasets: {len(train_files)} train, {len(val_files)} val files")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            persistent_workers=self.num_workers > 0
        )
    
    def _collate_fn(self, batch: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        # Filter out None examples
        batch = [item for item in batch if item is not None]
        
        if not batch:
            # Return empty batch if all examples failed
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long)
            }
        
        # Pad sequences to same length
        max_len = max(item["input_ids"].size(0) for item in batch)
        
        input_ids = []
        attention_masks = []
        
        for item in batch:
            seq_len = item["input_ids"].size(0)
            padding_len = max_len - seq_len
            
            # Pad input_ids
            padded_input = torch.cat([
                item["input_ids"],
                torch.full((padding_len,), self.tokenizer.pad_id, dtype=torch.long)
            ])
            input_ids.append(padded_input)
            
            # Pad attention_mask
            padded_mask = torch.cat([
                item["attention_mask"],
                torch.zeros(padding_len, dtype=torch.long)
            ])
            attention_masks.append(padded_mask)
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks)
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Decoder-Only Baseline for Music Generation")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing MIDI files")
    parser.add_argument("--max_len", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--snippet_length", type=int, default=256,
                       help="Length of each snippet in prompt")
    parser.add_argument("--num_snippets", type=int, default=2,
                       help="Number of snippets to include in prompt")
    
    # Model arguments
    parser.add_argument("--dim", type=int, default=1536,
                       help="Model dimension")
    parser.add_argument("--depth", type=int, default=16,
                       help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=24,
                       help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=-1,
                       help="Number of GPUs to use (-1 for all)")
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_true",
                       help="Training strategy")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       help="Training precision")
    
    # Logging arguments
    parser.add_argument("--project_name", type=str, default="staria-decoder-baseline",
                       help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Wandb run name")
    parser.add_argument("--log_dir", type=str, default="logs/decoder_baseline",
                       help="Logging directory")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/decoder_baseline",
                       help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set random seeds
    pl.seed_everything(42, workers=True)
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = MusicTokenizerWithStyle()
    
    # Create data module
    logger.info("Setting up data module...")
    data_module = PromptedMIDIDataModule(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        snippet_length=args.snippet_length,
        num_snippets=args.num_snippets
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = DecoderOnlyBaseline(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        max_len=args.max_len,
        lr=args.lr,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id
    )
    
    # Setup logging
    logger_instance = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        log_model="all"
    ) if args.project_name else TensorBoardLogger(args.log_dir)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="decoder-baseline-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus != 0 else "cpu",
        devices=args.gpus if args.gpus != -1 else "auto",
        strategy=args.strategy if args.gpus != 0 else None,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=logger_instance,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=50,
        val_check_interval=1.0,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from
    )
    
    # Save final model
    final_checkpoint = os.path.join(args.checkpoint_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_checkpoint)
    logger.info(f"Training completed. Final model saved to {final_checkpoint}")


if __name__ == "__main__":
    main()