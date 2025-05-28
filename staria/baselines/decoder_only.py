"""
Decoder-Only Baseline for Structured Music Generation

This baseline implements a pure decoder-only transformer that learns to generate
structured music by training on sequences that contain snippet prompts followed
by the full MIDI sequence.

Training format:
<PROMPT_START> <A_SECTION> [snippet_A] <B_SECTION> [snippet_B] <PROMPT_END> [full_midi_sequence]

The model learns to generate the full sequence conditioned on the snippet prompts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper


class DecoderOnlyBaseline(pl.LightningModule):
    """
    Pure decoder-only transformer for structured music generation.
    
    Unlike the full Staria model, this baseline uses a single transformer
    decoder that learns to generate music conditioned on snippet prompts
    that are prepended to the training sequences.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 1536,
        depth: int = 16,
        heads: int = 24,
        max_len: int = 4096,
        lr: float = 3e-4,
        pad_id: int = 2,
        bos_id: int = 1,
        eos_id: int = 3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_len = max_len
        self.lr = lr
        
        # Special token IDs
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        # Build decoder-only transformer
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_len,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=0.1,
                ff_dropout=0.1,
                cross_attend=False,  # Pure decoder, no cross-attention
                causal=True,
            )
        )
        
        # Wrap with autoregressive capabilities
        self.autoregressive_wrapper = AutoregressiveWrapper(
            self.transformer,
            ignore_index=pad_id,
            pad_value=pad_id
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        
    def _shift_right(self, x: torch.Tensor) -> torch.Tensor:
        """Shift tensor right by one position for decoder training."""
        shifted = torch.full_like(x, self.pad_id)
        shifted[:, 0] = self.bos_id
        shifted[:, 1:] = x[:, :-1].clone()
        return shifted
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the transformer."""
        return self.transformer(input_ids, **kwargs)
    
    def _compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss using teacher forcing.
        
        The input contains the full sequence: prompt + target music.
        We train the model to predict the next token at each position.
        """
        # Shift inputs for teacher forcing
        decoder_input = self._shift_right(input_ids)
        
        # Forward pass
        logits = self.transformer(decoder_input)
        
        # Compute loss against target tokens
        loss = self.loss_fn(
            logits.view(-1, self.vocab_size),
            input_ids.view(-1)
        )
        
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with prompt + sequence learning."""
        
        # The batch should contain sequences formatted as:
        # [prompt_tokens] + [full_music_sequence]
        input_ids = batch["input_ids"]
        
        # Compute loss
        loss = self._compute_loss(input_ids)
        
        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_ids = batch["input_ids"]
        loss = self._compute_loss(input_ids)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate music sequence conditioned on snippet prompts.
        
        Args:
            prompt_ids: Tensor of shape (batch_size, prompt_length) containing
                       the snippet prompts (e.g., <PROMPT_START> <A_SECTION> ...)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated sequence including the prompt
        """
        self.eval()
        
        # Use autoregressive wrapper for generation
        generated = self.autoregressive_wrapper.generate(
            prompt_ids,
            seq_len=max_new_tokens,
            temperature=temperature,
            filter_logits_fn=self._create_filter_fn(top_k, top_p),
            **kwargs
        )
        
        return generated
    
    def _create_filter_fn(self, top_k: Optional[int], top_p: Optional[float]):
        """Create logits filtering function for sampling."""
        def filter_fn(logits, **kwargs):
            if top_k is not None:
                logits = self._top_k_filter(logits, top_k)
            if top_p is not None:
                logits = self._top_p_filter(logits, top_p)
            return logits
        
        return filter_fn if (top_k is not None or top_p is not None) else None
    
    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k_logits, _ = torch.topk(logits, k)
        min_k_logits = top_k_logits[..., -1].unsqueeze(-1)
        return torch.where(logits < min_k_logits, torch.full_like(logits, float('-inf')), logits)
    
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits


class PromptedMIDIDataset:
    """
    Dataset class for training decoder-only baseline with prompted sequences.
    
    This dataset takes MIDI files and creates training examples where each
    sequence contains snippet prompts followed by the full MIDI sequence.
    """
    
    def __init__(
        self,
        midi_paths: List[str],
        tokenizer,
        max_len: int = 4096,
        snippet_length: int = 256,
        num_snippets: int = 2,
    ):
        self.midi_paths = midi_paths
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.snippet_length = snippet_length
        self.num_snippets = num_snippets
        
    def create_prompted_sequence(self, midi_tokens: List[str]) -> List[str]:
        """
        Create a training sequence with snippet prompts.
        
        Format: <PROMPT_START> <A_SECTION> [snippet_A] <B_SECTION> [snippet_B] <PROMPT_END> [full_sequence]
        """
        if len(midi_tokens) < self.snippet_length * self.num_snippets:
            return None  # Skip sequences that are too short
        
        # Extract snippets from different parts of the sequence
        snippet_positions = []
        total_length = len(midi_tokens)
        
        for i in range(self.num_snippets):
            # Distribute snippets across the sequence
            start_pos = (i * total_length) // self.num_snippets
            end_pos = min(start_pos + self.snippet_length, total_length)
            snippet_positions.append((start_pos, end_pos))
        
        # Build prompt
        prompt_tokens = ["<PROMPT_START>"]
        
        section_tokens = ["<A_SECTION>", "<B_SECTION>", "<C_SECTION>", "<D_SECTION>"]
        
        for i, (start, end) in enumerate(snippet_positions):
            if i < len(section_tokens):
                prompt_tokens.append(section_tokens[i])
                prompt_tokens.extend(midi_tokens[start:end])
        
        prompt_tokens.append("<PROMPT_END>")
        
        # Combine prompt with full sequence
        full_sequence = prompt_tokens + midi_tokens
        
        # Truncate if too long
        if len(full_sequence) > self.max_len:
            full_sequence = full_sequence[:self.max_len]
        
        return full_sequence
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get a training example."""
        midi_path = self.midi_paths[idx]
        
        try:
            # Tokenize MIDI file
            midi_tokens = self.tokenizer.tokenize_from_file(midi_path)
            if midi_tokens is None:
                return None
            
            # Create prompted sequence
            prompted_sequence = self.create_prompted_sequence(midi_tokens)
            if prompted_sequence is None:
                return None
            
            # Encode to IDs
            token_ids = self.tokenizer.encode(prompted_sequence)
            
            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(token_ids), dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.midi_paths)