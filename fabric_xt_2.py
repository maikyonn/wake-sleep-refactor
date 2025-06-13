#!/usr/bin/env python
"""
Train Staria music‑generation model with **Lightning Trainer** and modern callbacks.

Key features added:
- **TQDMProgressBar** for rich per‑epoch progress bars.
- **Stochastic Weight Averaging** (starting at epoch 10).
- **ModelCheckpoint** keeps only the single best model (lowest `val_loss`) and the last checkpoint.
- **DeviceStatsMonitor** logs real‑time GPU/CPU and memory stats to your loggers.
- **Perplexity tracking** for both training and validation.

The script no longer uses a manual Fabric training loop—everything is delegated to
`lightning.Trainer`, which automatically invokes the callbacks.
"""

from __future__ import annotations
import os, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from x_transformers import XTransformer
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    TQDMProgressBar,
    LearningRateMonitor,
    Callback,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import argparse
import numpy as np


from src.StariaTokenizer import (
    MusicTokenizerWithStyle,
    PROMPT_START_TOKEN, PROMPT_END_TOKEN,
    A_SECTION_TOKEN, B_SECTION_TOKEN, C_SECTION_TOKEN, D_SECTION_TOKEN,
    FORM_LABEL_MAP, IGNORE_LABEL_IDX, SPECIAL_TOKENS
)
from src.utils_new import music_style_from_labels

# Setup
L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)

_SECTION_TOK = {"A": A_SECTION_TOKEN, "B": B_SECTION_TOKEN, "C": C_SECTION_TOKEN, "D": D_SECTION_TOKEN}
_SNIPPET_LEN = 256

def get_style_labels_from_file(label_path: str) -> Optional[List[str]]:
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()
        if not content:
            return None
        return list(content)
    except Exception as e:
        logger.error(f"Error processing style file {label_path}: {e}", exc_info=False)
        return None

class OnDemandMidiDataset(Dataset):
    def __init__(self, dataset_path: str = None, pkl_file: str = None, tokenizer: MusicTokenizerWithStyle = None, max_len: int = 4096, use_snippet: bool = False, max_dataset_length: Optional[int] = None):
        self.tok = tokenizer
        self.max_len = max_len
        self.use_snippet = use_snippet
        self.max_dataset_length = max_dataset_length
        self.pad_id = tokenizer.pad_id

        # Load from pkl file if provided, otherwise use directory scanning
        if pkl_file:
            logger.info(f"Loading dataset from pkl file: {pkl_file}")
            self._load_from_pkl(pkl_file)
        elif dataset_path:
            logger.info(f"Loading dataset from directory: {dataset_path}")
            self._load_from_directory(dataset_path)
        else:
            raise ValueError("Either dataset_path or pkl_file must be provided")

    def _load_from_pkl(self, pkl_file: str):
        """Load dataset from preprocessed pkl file."""
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"PKL file not found: {pkl_file}")
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        if "data_records" not in data:
            raise ValueError("PKL file does not contain 'data_records' key")
        
        records = data["data_records"]
        metadata = data.get("metadata", {})
        
        logger.info(f"Loaded {len(records)} records from PKL file")
        logger.info(f"PKL metadata: {metadata}")
        
        # Convert records to file lists
        self.midi_files = []
        self.style_files = []
        
        for record in records:
            if "midi_file_path" not in record:
                logger.warning(f"Record missing midi_file_path: {record}")
                continue
            
            self.midi_files.append(record["midi_file_path"])
            
            # Handle style files for synthetic mode
            if "style_file_path" in record:
                self.style_files.append(record["style_file_path"])
            else:
                self.style_files.append(None)  # Real mode - no style files
        
        # Apply max_dataset_length limit
        if self.max_dataset_length is not None and self.max_dataset_length > 0:
            self.midi_files = self.midi_files[:self.max_dataset_length]
            self.style_files = self.style_files[:self.max_dataset_length]
            logger.info(f"Limited dataset to {len(self.midi_files)} files (max_dataset_length={self.max_dataset_length})")
        
        logger.info(f"Final dataset size: {len(self.midi_files)} files")

    def _load_from_directory(self, dataset_path: str):
        """Load dataset from directory structure (original behavior)."""
        self.dataset_path = dataset_path
        
        self.midi_files = []
        self.style_files = []
        midi_dir_path_obj = Path(dataset_path) / "midi"
        
        if midi_dir_path_obj.is_dir():
            midi_extensions = ['.mid', '.midi']
            for ext in midi_extensions:
                midi_paths = list(midi_dir_path_obj.rglob(f'*{ext}'))
                self.midi_files.extend([str(f) for f in midi_paths])
            
            # Create corresponding style file paths
            for midi_file in self.midi_files:
                midi_file_name_stem = Path(midi_file).stem
                style_file_name = f"{midi_file_name_stem}.txt"
                style_file_path = str(Path(dataset_path) / "style" / style_file_name)
                self.style_files.append(style_file_path)
            
            if self.max_dataset_length is not None and self.max_dataset_length > 0:
                self.midi_files = self.midi_files[:self.max_dataset_length]
                self.style_files = self.style_files[:self.max_dataset_length]
                logger.info(f"Limited dataset to {len(self.midi_files)} files (max_dataset_length={self.max_dataset_length})")
            
            logger.info(f"Found {len(self.midi_files)} MIDI files in directory {midi_dir_path_obj}")
        else:
            logger.error(f"Dataset path {dataset_path} does not contain a 'midi' subdirectory or {midi_dir_path_obj} is not a valid directory")
            self.midi_files = []
            self.style_files = []

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx >= len(self.midi_files):
            raise IndexError("Index out of bounds")
        
        midi_file_path_str = self.midi_files[idx]
        style_file_path_str = self.style_files[idx] if idx < len(self.style_files) else None

        if not os.path.exists(midi_file_path_str):
            logger.warning(f"MIDI file path does not exist: {midi_file_path_str} for index {idx}. Skipping.")
            return None

        tokens_str_list = self.tok.tokenize_from_file(midi_file_path_str)
        if tokens_str_list is None or not tokens_str_list:
            return None

        # Handle style files (synthetic mode) or skip style processing (real mode)
        if style_file_path_str and os.path.exists(style_file_path_str):
            style_labels_str_list = get_style_labels_from_file(style_file_path_str)
            if style_labels_str_list is None:
                return None
                
            if len(tokens_str_list) != len(style_labels_str_list):
                return None
        else:
            # Real mode - no style processing, create dummy style labels
            style_labels_str_list = ['A'] * len(tokens_str_list)

        if not self.use_snippet:
            ids_list = self.tok.encode(tokens_str_list)
            if len(ids_list) > self.max_len:
                ids_list = ids_list[:self.max_len]
            if not ids_list:
                return None
            return {"input_ids": torch.tensor(ids_list, dtype=torch.long),
                    "attention_mask": torch.ones(len(ids_list), dtype=torch.long)}
        else:
            tokens_for_snippet = self.tok.remove_instrument_prefix(tokens_str_list)

            music_style = music_style_from_labels(style_labels_str_list)
            enc_prompt_tokens = [PROMPT_START_TOKEN]
            runs = []
            start_idx = 0
            for i in range(1, len(style_labels_str_list)):
                if style_labels_str_list[i] != style_labels_str_list[i-1]:
                    runs.append((style_labels_str_list[i-1], start_idx, i-1))
                    start_idx = i
            if style_labels_str_list:
                runs.append((style_labels_str_list[-1], start_idx, len(style_labels_str_list)-1))
            else:
                return None

            for style_char in music_style:
                for lab_r, s_r, e_r in runs:
                    if lab_r == style_char:
                        enc_prompt_tokens.append(_SECTION_TOK[lab_r])
                        seg_end = min(e_r + 1, s_r + _SNIPPET_LEN)
                        if s_r < len(tokens_for_snippet):
                            seg = tokens_for_snippet[s_r : min(seg_end, len(tokens_for_snippet))]
                            enc_prompt_tokens.extend(seg)
                        break
            enc_prompt_tokens.append(PROMPT_END_TOKEN)
            
            enc_ids_list = self.tok.encode(enc_prompt_tokens)
            dec_target_tokens_no_prefix = tokens_str_list
            dec_ids_list = self.tok.encode(dec_target_tokens_no_prefix)

            for id_list_ref_wrapper in [[enc_ids_list], [dec_ids_list]]:
                current_list = id_list_ref_wrapper[0]
                if len(current_list) > self.max_len:
                    id_list_ref_wrapper[0] = current_list[:self.max_len]
                if not id_list_ref_wrapper[0]:
                    return None
            enc_ids_list, dec_ids_list = enc_ids_list, dec_ids_list

            return {
                "encoder_ids": torch.tensor(enc_ids_list, dtype=torch.long),
                "encoder_mask": torch.ones(len(enc_ids_list), dtype=torch.long),
                "decoder_ids": torch.tensor(dec_ids_list, dtype=torch.long),
                "decoder_mask": torch.ones(len(dec_ids_list), dtype=torch.long)
            }
        
def midi_collate_mapstyle(batch: List[Optional[Dict[str, Any]]], tokenizer: MusicTokenizerWithStyle) -> Optional[Dict[str, Any]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    encoder_ids = [b['encoder_ids'] for b in batch]
    decoder_ids = [b['decoder_ids'] for b in batch]
    encoder_mask = [b['encoder_mask'] for b in batch]
    decoder_mask = [b['decoder_mask'] for b in batch]
    
    return {
        'encoder_ids': pad_sequence(encoder_ids, batch_first=True, padding_value=tokenizer.pad_id),
        'decoder_ids': pad_sequence(decoder_ids, batch_first=True, padding_value=tokenizer.pad_id),
        'encoder_mask': pad_sequence(encoder_mask, batch_first=True, padding_value=0),
        'decoder_mask': pad_sequence(decoder_mask, batch_first=True, padding_value=0)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Custom Callback for Advanced LR Scheduling with SWA
# ──────────────────────────────────────────────────────────────────────────────
class LRSchedulerWithSWACallback(Callback):
    """Custom callback to coordinate LR scheduling with SWA."""
    
    def __init__(self, swa_start_epoch_ratio=0.75):
        self.swa_start_epoch_ratio = swa_start_epoch_ratio
        self.swa_started = False
    
    def on_train_epoch_start(self, trainer, pl_module):
        # Check if SWA should start
        current_epoch_ratio = trainer.current_epoch / trainer.max_epochs
        
        if current_epoch_ratio >= self.swa_start_epoch_ratio and not self.swa_started:
            self.swa_started = True
            logger.info(f"SWA period started at epoch {trainer.current_epoch}")
            
            # Log the transition
            pl_module.log("swa_active", 1.0, on_epoch=True, logger=True)
        
        # Log current learning rate
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        pl_module.log("learning_rate_manual", current_lr, on_epoch=True, logger=True)


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule with Perplexity Support
# ──────────────────────────────────────────────────────────────────────────────
class LitXTransformer(L.LightningModule):
    def __init__(
        self,
        tokenizer: MusicTokenizerWithStyle,
        dim: int = 1536,
        enc_depth: int = 6,
        enc_heads: int = 8,
        dec_depth: int = 12,
        dec_heads: int = 16,
        max_seq_len: int = 4096,
        lr: float = 1e-3,
        attn_flash: bool = True,
        rotary_pos_emb: bool = True,
        dropout: float = 0.2,
        dec_cross_residual_attn: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.lr = lr
        
        vocab_size = tokenizer.vocab_size
        
        self.model = XTransformer(
            dim=dim,
            enc_num_tokens=vocab_size,
            enc_depth=enc_depth,
            enc_heads=enc_heads,
            enc_max_seq_len=max_seq_len,
            dec_num_tokens=vocab_size,
            dec_depth=dec_depth,
            dec_heads=dec_heads,
            dec_max_seq_len=max_seq_len,
            tie_token_emb=True,
            attn_flash=attn_flash,
            rotary_pos_emb=rotary_pos_emb,
            dec_cross_residual_attn=dec_cross_residual_attn,
            dec_attn_dropout=dropout,
            dec_ff_dropout=dropout
        )

    def forward(self, encoder_ids, decoder_ids, mask=None):
        return self.model(encoder_ids, decoder_ids, mask=mask)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
            
        encoder_ids = batch['encoder_ids']
        decoder_ids = batch['decoder_ids']
        encoder_mask = batch['encoder_mask'].bool()
        
        loss = self.model(encoder_ids, decoder_ids, mask=encoder_mask)
        
        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate and log perplexity
        perplexity = torch.exp(loss)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
            
        encoder_ids = batch['encoder_ids']
        decoder_ids = batch['decoder_ids']
        encoder_mask = batch['encoder_mask'].bool()
        
        loss = self.model(encoder_ids, decoder_ids, mask=encoder_mask)
        
        # Log loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Calculate and log perplexity
        perplexity = torch.exp(loss)
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            betas=(0.9, 0.95), 
            weight_decay=0.1
        )
        
        # Return just the optimizer without any scheduler
        return optimizer


def custom_generate_top_p(model, encoder_ids, decoder_start, max_length=256, encoder_mask=None, use_tqdm=False, tokenizer=None, top_p=0.9, temperature=1.0):
    """
    Custom generation function using top-p (nucleus) sampling for token selection.
    
    Args:
        model: The XTransformer model
        encoder_ids: Encoder input tokens [batch_size, enc_seq_len]
        decoder_start: Starting decoder token [batch_size, 1]
        max_length: Maximum generation length
        encoder_mask: Encoder attention mask [batch_size, enc_seq_len]
        use_tqdm: Whether to show progress bar during generation
        tokenizer: Tokenizer instance to get eos_id for early stopping
        top_p: Nucleus sampling parameter (default: 0.9)
        temperature: Temperature for softmax scaling (default: 1.0)
    
    Returns:
        Generated sequence [batch_size, generated_length]
    """
    device = encoder_ids.device
    batch_size = encoder_ids.size(0)
    
    # Get end token ID for early stopping
    eos_id = tokenizer.eos_id if tokenizer and tokenizer.eos_id is not None else None
    
    # First, encode the input sequence
    with torch.no_grad():
        # Get encoder embeddings
        encodings = model.encoder(encoder_ids, mask=encoder_mask, return_embeddings=True)
    
    # Initialize the decoder sequence with the start token
    decoder_ids = decoder_start.clone()  # [batch_size, 1]
    
    # Set up progress bar if requested
    progress_bar = tqdm(range(max_length - 1), desc="Generating tokens") if use_tqdm else range(max_length - 1)
    
    with torch.no_grad():
        for step in progress_bar:  # -1 because we already have the start token
            try:
                # Forward pass through the decoder only
                # We need to access the decoder's transformer directly to get logits
                decoder_wrapper = model.decoder
                
                # Get logits from the decoder
                logits = decoder_wrapper.net(decoder_ids, context=encodings, context_mask=encoder_mask)
                
                # Get the logits for the last position
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a mask for the original logits tensor and set filtered tokens to -inf
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                
                # Append the new token to the sequence
                decoder_ids = torch.cat([decoder_ids, next_token], dim=1)
                
                # Update progress bar description if using tqdm
                if use_tqdm:
                    progress_bar.set_postfix({"current_length": decoder_ids.size(1)})
                
                # Stop if we generate an end token (check all samples in batch)
                if eos_id is not None and torch.any(next_token == eos_id):
                    if use_tqdm:
                        progress_bar.set_description("Generation stopped - <E> token generated")
                        progress_bar.close()
                    break
                    
            except Exception as e:
                logger.error(f"Error during generation step {step}: {e}")
                if use_tqdm:
                    progress_bar.close()
                break
    
    if use_tqdm and hasattr(progress_bar, 'close'):
        progress_bar.close()
    
    return decoder_ids


def inference_on_sample(model, sample, tokenizer, max_length=4096, use_tqdm=False):
    """Perform inference on a single sample using Lightning model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure model is on the correct device
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        encoder_ids = sample['encoder_ids'].unsqueeze(0).to(device)  # Add batch dimension
        encoder_mask = sample['encoder_mask'].unsqueeze(0).to(device).bool()
        
        # Create a starting sequence for the decoder (single token)
        seq_out_start = torch.tensor([[6]], device=device, dtype=torch.long)  # Shape: [1, 1]
        
        # Access the underlying XTransformer model from Lightning module
        underlying_model = model.model if hasattr(model, 'model') else model
        
        # Use our custom generation function
        generated = custom_generate_top_p(
            model=underlying_model,
            encoder_ids=encoder_ids,
            decoder_start=seq_out_start,
            max_length=max_length,
            encoder_mask=encoder_mask,
            use_tqdm=use_tqdm,
            tokenizer=tokenizer
        )
        
        # Remove batch dimension and move to CPU for decoding
        generated = generated.squeeze(0).cpu()
        
        # Decode the generated tokens
        generated_tokens = tokenizer.decode(generated.tolist())
        
        return generated_tokens


def make_dataloaders(
    train_pkl_file: str,
    val_pkl_file: str,
    tokenizer: MusicTokenizerWithStyle,
    batch_size: int = 1,
    num_workers: int = 4,
    max_len: int = 4096,
    use_snippet: bool = True,
):
    """Create train and validation dataloaders."""
    
    def collate_fn(batch):
        return midi_collate_mapstyle(batch, tokenizer)
    
    train_dataset = OnDemandMidiDataset(
        pkl_file=train_pkl_file,
        tokenizer=tokenizer,
        max_len=max_len,
        use_snippet=use_snippet
    )
    
    val_dataset = OnDemandMidiDataset(
        pkl_file=val_pkl_file,
        tokenizer=tokenizer,
        max_len=max_len,
        use_snippet=use_snippet
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True
    )
    
    return train_dataloader, val_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train music transformer model with Lightning")
    parser.add_argument("--train_pkl_file", type=str, required=True, help="Path to preprocessed pkl file containing training dataset paths")
    parser.add_argument("--val_pkl_file", type=str, required=True, help="Path to preprocessed pkl file containing validation dataset paths")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/staria_xt_2", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # SWA arguments
    parser.add_argument("--swa_start_ratio", type=float, default=0.75,
                       help="Start SWA at this ratio of total epochs")
    parser.add_argument("--swa_lr_ratio", type=float, default=0.05,
                       help="SWA learning rate as ratio of base LR")
    
    return parser.parse_args()


def main():
    """Main function with Lightning Trainer and modern callbacks."""
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = MusicTokenizerWithStyle()
    
    # Create dataloaders
    train_dataloader, val_dataloader = make_dataloaders(
        args.train_pkl_file,
        args.val_pkl_file,
        tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_snippet=True
    )
    
    logger.info(f"Training dataset: {len(train_dataloader.dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataloader.dataset)} samples")
    
    # Create model with fixed parameters
    model = LitXTransformer(
        tokenizer=tokenizer,
        lr=args.lr,
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup loggers
    tb_logger = TensorBoardLogger("logs", name="music_transformer")
    wandb_logger = WandbLogger(
        project="music-transformer",
        name=f"train_lr{args.lr}_bs{args.batch_size}_devices{args.devices}"
    )
    
    # Setup callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        StochasticWeightAveraging(swa_epoch_start=10, swa_lrs=args.lr * args.swa_lr_ratio),
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        DeviceStatsMonitor(),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # Setup trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision="bf16-mixed",
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        strategy="ddp" if args.devices > 1 else "auto",
        callbacks=callbacks,
        logger=[tb_logger, wandb_logger],
        enable_progress_bar=True,
        log_every_n_steps=10,
        accumulate_grad_batches=16,
    )
    
    # Train the model
    logger.info("Starting training with Lightning Trainer...")
    trainer.fit(
        model, 
        train_dataloader, 
        val_dataloader, 
        ckpt_path=args.resume_from_checkpoint
    )
    
    logger.info("Training complete!")
    if trainer.checkpoint_callback.best_model_path:
        logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    
    # Perform post-training inference with perplexity analysis
    if len(train_dataloader.dataset) > 0:
        logger.info("Performing post-training inference on a sample...")
        
        # Load best model for inference
        if trainer.checkpoint_callback.best_model_path:
            best_model = LitXTransformer.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
                tokenizer=tokenizer
            )
        else:
            best_model = model
        
        # Get a sample from the training dataset
        sample = None
        for i in range(len(train_dataloader.dataset)):
            sample = train_dataloader.dataset[i]
            if sample is not None:
                break
        
        if sample is not None:
            try:
                best_model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                best_model.to(device)
                
                # Extract and decode tokens
                encoder_ids = sample['encoder_ids'].tolist()
                decoder_ids = sample['decoder_ids'].tolist()
                
                encoder_tokens = tokenizer.decode(encoder_ids)
                target_tokens = tokenizer.decode(decoder_ids)
                
                logger.info(f"Running inference on sample with encoder length: {len(encoder_ids)}, target length: {len(decoder_ids)}")
                
                # Generate output
                generated_tokens = inference_on_sample(best_model, sample, tokenizer, max_length=1024, use_tqdm=True)
                
                # Calculate perplexity on the sample
                with torch.no_grad():
                    batch = {
                        'encoder_ids': sample['encoder_ids'].unsqueeze(0).to(device),
                        'decoder_ids': sample['decoder_ids'].unsqueeze(0).to(device),
                        'encoder_mask': sample['encoder_mask'].unsqueeze(0).to(device),
                        'decoder_mask': sample['decoder_mask'].unsqueeze(0).to(device)
                    }
                    
                    loss = best_model.validation_step(batch, 0)
                    if loss is not None:
                        sample_perplexity = torch.exp(loss).item()
                    else:
                        sample_perplexity = float('nan')
                
                # Save results to file
                results_file = "post_training_inference_results.txt"
                with open(results_file, 'w') as f:
                    f.write("POST-TRAINING INFERENCE RESULTS (Lightning Trainer with Perplexity)\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write("SAMPLE PERPLEXITY:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Loss: {loss.item() if loss is not None else 'N/A':.4f}\n")
                    f.write(f"Perplexity: {sample_perplexity:.2f}\n\n")
                    
                    f.write("ENCODER TOKENS (INPUT):\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Length: {len(encoder_tokens)}\n")
                    f.write(f"Tokens: {encoder_tokens}\n\n")
                    
                    f.write("TARGET TOKENS (EXPECTED OUTPUT):\n")
                    f.write("-" * 35 + "\n")
                    f.write(f"Length: {len(target_tokens)}\n")
                    f.write(f"Tokens: {target_tokens}\n\n")
                    
                    f.write("GENERATED TOKENS (ACTUAL OUTPUT):\n")
                    f.write("-" * 35 + "\n")
                    f.write(f"Length: {len(generated_tokens) if generated_tokens else 0}\n")
                    f.write(f"Tokens: {generated_tokens if generated_tokens else 'None'}\n\n")
                    
                    # Calculate and save match statistics
                    if generated_tokens and target_tokens:
                        match_count = sum(1 for g, t in zip(generated_tokens, target_tokens) if g == t)
                        match_ratio = match_count / min(len(generated_tokens), len(target_tokens))
                        
                        f.write("COMPARISON STATISTICS:\n")
                        f.write("-" * 25 + "\n")
                        f.write(f"Matching tokens: {match_count}\n")
                        f.write(f"Comparison length: {min(len(generated_tokens), len(target_tokens))}\n")
                        f.write(f"Match ratio: {match_ratio:.3f} ({match_ratio*100:.1f}%)\n")
                        f.write(f"Generated/Target length ratio: {len(generated_tokens)/len(target_tokens):.3f}\n\n")
                    
                    f.write("RAW TOKEN IDS:\n")
                    f.write("-" * 15 + "\n")
                    f.write(f"Encoder IDs: {encoder_ids[:50]}{'...' if len(encoder_ids) > 50 else ''}\n")
                    f.write(f"Target IDs: {decoder_ids[:50]}{'...' if len(decoder_ids) > 50 else ''}\n")
                    if generated_tokens:
                        generated_ids = tokenizer.encode(generated_tokens)
                        f.write(f"Generated IDs: {generated_ids[:50]}{'...' if len(generated_ids) > 50 else ''}\n")
                    
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("PERPLEXITY INTERPRETATION:\n")
                    f.write("-" * 25 + "\n")
                    if sample_perplexity < 10:
                        f.write(f"Perplexity {sample_perplexity:.2f}: Model is very confident (strong patterns)\n")
                    elif sample_perplexity < 30:
                        f.write(f"Perplexity {sample_perplexity:.2f}: Model has normal confidence (typical musical flow)\n")
                    elif sample_perplexity < 100:
                        f.write(f"Perplexity {sample_perplexity:.2f}: Model is somewhat uncertain (complex/transitional section)\n")
                    else:
                        f.write(f"Perplexity {sample_perplexity:.2f}: Model is very uncertain (struggling with prediction)\n")
                
                logger.info(f"Inference results saved to: {results_file}")
                logger.info(f"Generated {len(generated_tokens) if generated_tokens else 0} tokens")
                logger.info(f"Sample perplexity: {sample_perplexity:.2f}")
                
                if generated_tokens and target_tokens:
                    match_count = sum(1 for g, t in zip(generated_tokens, target_tokens) if g == t)
                    match_ratio = match_count / min(len(generated_tokens), len(target_tokens))
                    logger.info(f"Token match ratio: {match_ratio:.3f} ({match_count}/{min(len(generated_tokens), len(target_tokens))})")
                
            except Exception as e:
                logger.error(f"Error during post-training inference: {e}")
                # Save error to file
                with open("post_training_inference_results.txt", 'w') as f:
                    f.write("POST-TRAINING INFERENCE RESULTS (Lightning Trainer with Perplexity)\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"ERROR DURING INFERENCE: {str(e)}\n")
        else:
            logger.warning("No valid sample found for post-training inference")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()