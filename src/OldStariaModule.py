r"""lightning_models.py
=======================
⚡ **PyTorch‑Lightning wrappers** for the `simple_transformer` models you now
have in the canvas.  Three classes are provided:

* **StructureEncoder**        – trains the encoder alone with Masked‑LM (BERT‑style)
* **DecoderLM**        – trains the decoder alone with causal LM (GPT‑style)
* **Seq2SeqLM**        – loads both trained weights, toggles cross‑attention,
                         and fine‑tunes end‑to‑end on sequence‑to‑sequence tasks

All three share the same conveniences:
--------------------------------------
* AdamW + linear warm‑up scheduler (configurable)
* Padding index & vocab size read from the passed tokenizer
* Quick `.from_pretrained()` helpers load weights from `.ckpt` files produced
  by Lightning’s `ModelCheckpoint` callback.
* A `_step_common` util returns loss & accuracy so training/valid loops are
  terse and consistent.

Example
-------
```python
# 1) independent pre‑train -----------------------------------------------
enc_module = StructureEncoder(tokenizer, dim=512, depth=12, heads=8)
dec_module = DecoderLM(tokenizer, dim=512, depth=6,  heads=8)

# → fit(pl.Trainer(...)) each one separately, save checkpoints …

# 2) joint fine‑tune ------------------------------------------------------
seq2seq = Seq2SeqLM.from_pretrained(
    tokenizer,
    encoder_ckpt="ckpts/enc_epoch=2-step=8000.ckpt",
    decoder_ckpt="ckpts/dec_epoch=4-step=16000.ckpt",
    lr=5e-5,
)
trainer.fit(seq2seq, datamodule=your_dm)
```
"""
from __future__ import annotations
import math, random
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from src.StariaTokenizer import MusicTokenizerWithStyle
import torch
from x_transformers import TransformerWrapper, Decoder, Encoder

def _shift_right(x, pad_id, bos_id):
    """
    Shift tensor right by one position for decoder training.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len]
        pad_id: Padding token ID
        bos_id: Beginning of sequence token ID
        
    Returns:
        Tensor of same shape as input, shifted right with BOS token at start
    """
    # Create a tensor filled with pad_id of the same shape as x
    shifted = torch.full_like(x, pad_id)
    
    # Fill the first position with bos_id
    shifted[:, 0] = bos_id
    
    # Copy x to shifted, offset by 1 position
    shifted[:, 1:] = x[:, :-1].clone()
    
    return shifted

# ---------------------------------------------------------------------------
# Contrastive Encoder LightningModule
# ---------------------------------------------------------------------------
class ContrastiveStructureEncoder(pl.LightningModule):
    def __init__(
        self,
        tokenizer: MusicTokenizerWithStyle,
        dim: int = 1536,
        depth: int = 6,
        heads: int = 8,
        max_len: int = 4096,
        temp: float = 0.08,
        lambda_local: float = 0.7,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.temp = temp
        self.lambda_local = lambda_local
        self.lr = lr
        # encoder model
        self.model = TransformerWrapper(
            num_tokens = tokenizer.vocab_size,
            max_seq_len = max_len,
            attn_layers = Encoder(
                dim = dim,
                depth = depth,
                heads = heads
            )
        )
        # save hyperparameters
        self.save_hyperparameters('dim','depth','heads','max_len','temp','lambda_local','lr')

    def forward(self, input_ids, mask, return_hiddens=False):
        # key_padding_mask: True for positions to mask
        h = self.model(input_ids, mask=mask, return_hiddens)  # [B, T, D]
        if return_hiddens:
            return h
        # mean-pool non-padded positions
        mask_f = mask.unsqueeze(-1)
        summed = (h * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1)
        z = summed / counts
        # normalize
        return F.normalize(z, dim=-1)

    def training_step(self, batch, batch_idx):
        # local
        z1 = self(batch['x1_local'], batch['mask_local'])
        z2 = self(batch['x2_local'], batch['mask_local'])
        # prompt
        p1 = self(batch['x1_prompt'], batch['mask_prompt'])
        p2 = self(batch['x2_prompt'], batch['mask_prompt'])
        loss_local = self.nt_xent(z1, z2)
        loss_prompt = self.nt_xent(p1, p2)
        loss = self.lambda_local * loss_local + (1 - self.lambda_local) * loss_prompt
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z1 = self(batch['x1_local'], batch['mask_local'])
        z2 = self(batch['x2_local'], batch['mask_local'])
        p1 = self(batch['x1_prompt'], batch['mask_prompt'])
        p2 = self(batch['x2_prompt'], batch['mask_prompt'])
        loss_local = self.nt_xent(z1, z2)
        loss_prompt = self.nt_xent(p1, p2)
        loss = self.lambda_local * loss_local + (1 - self.lambda_local) * loss_prompt
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

    def nt_xent(self, zi, zj):
        # batch dot / temperature
        batch_size = zi.size(0)
        z = torch.cat([zi, zj], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.T) / self.temp
        # mask out self-sim
        mask = (~torch.eye(2*batch_size, device=sim.device).bool()).float()
        exp = torch.exp(sim) * mask
        pos = torch.exp(torch.sum(zi * zj, dim=-1) / self.temp)
        pos = torch.cat([pos, pos], dim=0)
        denom = exp.sum(dim=1)
        loss = -torch.log(pos / denom).mean()
        return loss

# ---------------------------------------------------------------------------
# Decoder – Causal Language Model ------------------------------------------
# ---------------------------------------------------------------------------
class DecoderLM(pl.LightningModule):
    def __init__(
        self, 
        tokenizer, 
        dim=1536, 
        depth=16, 
        heads=24, 
        max_len=4096, 
        cross_attend=False, 
        lr: float = 1e-4, 
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.lr = lr
        self.pad_id = tokenizer.pad_id
        self.mask_id = tokenizer.mask_id if hasattr(tokenizer, "mask_id") else tokenizer.vocab_size - 1
        self.bos_id = getattr(tokenizer, "bos_id", self.mask_id)  # fall back to mask if none
        
        self.model = TransformerWrapper(
            num_tokens=tokenizer.vocab_size,
            max_seq_len=max_len,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                cross_attend=cross_attend,
                attn_flash = True
            )
        )
        self.cross_attend_enabled = cross_attend
        self.save_hyperparameters("lr", "dim", "depth", "heads", "max_len")

    # --------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

    # --------------------------------------------------
    def forward(self, input_ids, *, context=None):
        return self.model(input_ids, context=context)

    # --------------------------------------------------
    def _step_common(self, logits: torch.Tensor, targets: torch.Tensor, split: str):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.pad_id)
        preds = logits.argmax(-1)
        acc = (preds.eq(targets) & targets.ne(self.pad_id)).float().mean()
        self.log(f"{split}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{split}_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        if "input_ids" in batch:
            ids = batch["input_ids"]
        else:
            ids = batch["decoder_ids"]

        ids = ids.to("cuda")
        labels = ids.clone()
        dec_inp = _shift_right(ids, self.pad_id, self.bos_id)

        logits = self(dec_inp)
        return self._step_common(logits, labels, "train")

    def validation_step(self, batch, batch_idx):
        if "input_ids" in batch:
            ids = batch["input_ids"]
        else:
            ids = batch["decoder_ids"]

        ids = ids.to("cuda")
        labels = ids.clone()
        dec_inp = _shift_right(ids, self.pad_id, self.bos_id)
        logits = self(dec_inp)
        self._step_common(logits, labels, "val")
    # --------------------------------------------------
    def generate(self, prompt_ids, max_length=4096, temperature=1.0, top_k=0, top_p=0.9, 
                 context=None, output_path=None, debug=False):
        """
        Generate a sequence from a prompt using autoregressive sampling.
        
        Args:
            prompt_ids: Tensor of shape [batch_size, seq_len] containing prompt token ids
            max_length: Maximum length of the generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep (0 = disabled)
            top_p: Cumulative probability threshold for nucleus sampling
            context: Optional encoder context for cross-attention
            output_path: Optional path to save the generated MIDI file
            debug: Whether to print debugging information during generation
            
        Returns:
            Generated token ids of shape [batch_size, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_size = prompt_ids.shape[0]
        
        # Move inputs to the correct device
        prompt_ids = prompt_ids.to(device)
        if context is not None:
            context = context.to(device)
        
        # Start with the prompt
        generated = prompt_ids.clone()
        
        # Generate tokens one by one
        with torch.no_grad():
            for _ in range(max_length - prompt_ids.shape[1]):
                # Get model predictions
                logits = self(generated, context=context)  # context is None for DecoderLM usually
                
                # Only consider the last token's predictions
                next_token_logits = logits[:, -1, :]  # Get logits for the last position
                
                # --- BEGIN DEBUGGING BLOCK ---
                if debug:
                    current_probs = F.softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(current_probs, k=20)  # Look at top 20

                    print(f"\n--- Generating token {generated.size(1)} ---")
                    if generated.size(1) > 0:
                        last_generated_token_id = generated[0, -1].item()
                        last_generated_token_str = self.tokenizer.decode([last_generated_token_id])[0]
                        print(f"Last token generated: {last_generated_token_str} (ID: {last_generated_token_id})")

                    print("Top 20 next token predictions:")
                    for i in range(top_k_indices.size(1)):
                        token_id = top_k_indices[0, i].item()
                        token_str = self.tokenizer.decode([token_id])[0]
                        probability = top_k_probs[0, i].item()
                        print(f"  - Token: {token_str}, ID: {token_id}, Probability: {probability:.4f}")
                # --- END DEBUGGING BLOCK ---
                
                # Simply use argmax for next token selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append the sampled token to the sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Stop if we've generated an EOS token
                if (next_token == self.tokenizer.eos_id).any():
                    # Find the first EOS token in each sequence
                    eos_positions = (generated == self.tokenizer.eos_id).nonzero()
                    if len(eos_positions) > 0:
                        # Group by batch index
                        batch_indices = {}
                        for batch_idx, seq_pos in eos_positions:
                            if batch_idx.item() not in batch_indices or seq_pos < batch_indices[batch_idx.item()]:
                                batch_indices[batch_idx.item()] = seq_pos
                        
                        # If all sequences have an EOS token, stop generation
                        if len(batch_indices) == batch_size:
                            break
        
        print(self.tokenizer.decode(generated[0].cpu().tolist()))
        # Save to MIDI file if output path is provided
        if output_path is not None and batch_size == 1:
            self.tokenizer.ids_to_file(generated[0].cpu().tolist(), output_path)
            
        return generated

    # --------------------------------------------------
    def enable_cross_attention(self, enabled: bool = True):
        self.model.attn_layers.cross_attend = True

    # --------------------------------------------------
    @classmethod
    def from_pretrained(cls, tokenizer, ckpt_path: str, **override_kwargs):
        model = cls.load_from_checkpoint(ckpt_path, tokenizer=tokenizer, **override_kwargs)
        return model

# ---------------------------------------------------------------------------
# Seq2Seq joint fine‑tuning with staged unfreezing --------------------------------
# ---------------------------------------------------------------------------
class StariaModel(pl.LightningModule):
    def __init__(
        self,
        tokenizer,
        encoder=None,
        decoder=None,
        lr: float = 1e-5,
        stage: str = "A",  # "A", "B", or "C"
        decoder_top_unfrozen: int = 4,  # for stage B
        encoder_lr: float = 1e-5,       # for stage C
        decoder_top_lr: float = 1e-5,   # for stage B (10x lower than base)
        **kwargs,
    ):
        super().__init__()
        print("Initializing StariaModel")
        self.tokenizer = tokenizer
        self.lr = lr
        
        # Get special token IDs from tokenizer
        self.pad_id = getattr(tokenizer, 'pad_id', 0)
        self.bos_id = getattr(tokenizer, 'bos_id', 1)
        self.eos_id = getattr(tokenizer, 'eos_id', 2)

        # 1) Build encoder / decoder UNCONDITIONALLY
        self.encoder = ContrastiveStructureEncoder(tokenizer, dim=1536, depth=6, heads=8, max_len=4096, temp=0.08, lambda_local=0.7, lr=1e-3)
        self.decoder = DecoderLM(tokenizer, dim=1536, depth=16, heads=24, max_len=4096, cross_attend=True)
            
        self.stage = stage
        self.decoder_top_unfrozen = decoder_top_unfrozen
        self.encoder_lr = encoder_lr
        self.decoder_top_lr = decoder_top_lr
        
        # Apply freezing based on stage
        self._apply_stage_freezing()
            
    def _apply_stage_freezing(self):
        """
        Implements staged freezing/unfreezing for finetuning:
        Stage A: Only cross-attn projections, LayerNorms, and encoder-to-decoder adapter trainable.
        Stage B: Unfreeze top N decoder blocks (self-attn + FFN) with lower LR, rest frozen.
        Stage C: Unfreeze encoder with even lower LR.
        """
        # Check if encoder and decoder are initialized
        if self.encoder is None:
            print("Warning: Encoder is None. Cannot apply freezing to encoder.")
        if self.decoder is None:
            print("Warning: Decoder is None. Cannot apply freezing to decoder.")
            return
            
        # Count and print trainable parameters before applying freezing
        def count_trainable_params(module):
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
            
        encoder_params_before = count_trainable_params(self.encoder)
        decoder_params_before = count_trainable_params(self.decoder)
        total_params_before = encoder_params_before + decoder_params_before
        
        print(f"Before freezing - Trainable parameters:")
        print(f"  Encoder: {encoder_params_before:,}")
        print(f"  Decoder: {decoder_params_before:,}")
        print(f"  Total:   {total_params_before:,}")
        # Helper: recursively freeze all params
        def freeze_all(module):
            for p in module.parameters():
                p.requires_grad = False

        # Helper: unfreeze all params
        def unfreeze_all(module):
            for p in module.parameters():
                p.requires_grad = True

        # Helper: unfreeze LayerNorms
        def unfreeze_layernorms(module):
            for m in module.modules():
                if m.__class__.__name__.lower().startswith("layernorm"):
                    for p in m.parameters():
                        p.requires_grad = True

        # Helper: unfreeze cross-attn projections
        def unfreeze_cross_attn(module):
            # Check if module has the expected structure
            if hasattr(module, "attn_layers") and hasattr(module.attn_layers, "__iter__"):
                for block in module.attn_layers:
                    if hasattr(block, "cross_attn"):
                        for p in block.cross_attn.parameters():
                            p.requires_grad = True
            # Handle case where decoder has different structure
            elif hasattr(module, "cross_attn"):
                for p in module.cross_attn.parameters():
                    p.requires_grad = True

        # Helper: unfreeze encoder-to-decoder adapter if present
        def unfreeze_adapter(module):
            if hasattr(module, "adapter"):
                for p in module.adapter.parameters():
                    p.requires_grad = True

        # Helper: unfreeze top N decoder blocks (self-attn + FFN)
        def unfreeze_top_decoder_blocks(module, n):
                
            # Then selectively unfreeze top N blocks
            if hasattr(module, "attn_layers") and hasattr(module.attn_layers, "layers"):
                # Handle Decoder class from AriaTransformerModel
                total = len(module.attn_layers.layers)
                for i in range(total - n, total):
                    for p in module.attn_layers.layers[i].parameters():
                        p.requires_grad = True
            elif hasattr(module, "attn_layers") and isinstance(module.attn_layers, nn.ModuleList):
                # Handle case with direct ModuleList
                total = len(module.attn_layers)
                for i in range(total - n, total):
                    for p in module.attn_layers[i].parameters():
                        p.requires_grad = True
            # Handle case where decoder has different structure
            elif hasattr(module, "layers") and hasattr(module.layers, "__iter__"):
                total = len(module.layers)
                for i in range(total - n, total):
                    for p in module.layers[i].parameters():
                        p.requires_grad = True

        # Stage A: Only cross-attn, LayerNorms, and adapter trainable
        if self.stage == "A":
            print(f"Applying stage freezing for stage: {self.stage}")
            freeze_all(self.decoder)
            freeze_all(self.encoder)
            unfreeze_cross_attn(self.decoder)
            unfreeze_layernorms(self.decoder)
            unfreeze_adapter(self.decoder)
        # Stage B: Unfreeze top N decoder blocks (self-attn + FFN) with lower LR, rest frozen
        elif self.stage == "B":
            print(f"Applying stage freezing for stage: {self.stage}")
            freeze_all(self.decoder)
            freeze_all(self.encoder)
            unfreeze_cross_attn(self.decoder)
            unfreeze_layernorms(self.decoder)
            unfreeze_adapter(self.decoder)
            unfreeze_top_decoder_blocks(self.decoder, self.decoder_top_unfrozen)
        # Stage C: Unfreeze encoder with even lower LR
        elif self.stage == "C":
            print(f"Applying stage freezing for stage: {self.stage}")
            freeze_all(self.decoder)
            unfreeze_cross_attn(self.decoder)
            unfreeze_layernorms(self.decoder)
            unfreeze_adapter(self.decoder)
            unfreeze_top_decoder_blocks(self.decoder, self.decoder_top_unfrozen)
            unfreeze_all(self.encoder)
        else:
            print("Applying stage freezing for stage: default")
            # Default: unfreeze everything
            unfreeze_all(self.decoder)
            unfreeze_all(self.encoder)

    # --------------------------------------------------
    def configure_optimizers(self):
        """
        Custom optimizer setup for staged finetuning.
        - Stage A: Only cross-attn, LayerNorms, and adapter params.
        - Stage B: Add top decoder blocks with lower LR.
        - Stage C: Add encoder with even lower LR.
        """
        import torch.optim as optim

        # Collect parameter groups
        param_groups = []

        # Helper: get params by filter
        def get_params(module, filter_fn):
            params = []
            for n, m in module.named_modules():
                if filter_fn(m):
                    for p in m.parameters(recurse=False):
                        if p.requires_grad:
                            params.append(p)
            return params

        # Helper: get cross-attn params
        def cross_attn_params(module):
            ps = []
            if hasattr(module, "attn_layers") and hasattr(module.attn_layers, "__iter__"):
                for block in module.attn_layers:
                    if hasattr(block, "cross_attn"):
                        for p in block.cross_attn.parameters():
                            if p.requires_grad:
                                ps.append(p)
            elif hasattr(module, "cross_attn"):
                for p in module.cross_attn.parameters():
                    if p.requires_grad:
                        ps.append(p)
            return ps

        # Helper: get LayerNorm params
        def layernorm_params(module):
            ps = []
            for m in module.modules():
                if m.__class__.__name__.lower().startswith("layernorm"):
                    for p in m.parameters():
                        if p.requires_grad:
                            ps.append(p)
            return ps

        # Helper: get adapter params
        def adapter_params(module):
            ps = []
            if hasattr(module, "adapter"):
                for p in module.adapter.parameters():
                    if p.requires_grad:
                        ps.append(p)
            return ps

        # Helper: get top N decoder block params
        def top_decoder_block_params(module, n):
            ps = []
            if hasattr(module, "attn_layers") and hasattr(module.attn_layers, "__iter__"):
                total = len(module.attn_layers)
                for i in range(total - n, total):
                    for p in module.attn_layers[i].parameters():
                        if p.requires_grad:
                            ps.append(p)
            elif hasattr(module, "layers") and hasattr(module.layers, "__iter__"):
                total = len(module.layers)
                for i in range(total - n, total):
                    for p in module.layers[i].parameters():
                        if p.requires_grad:
                            ps.append(p)
            return ps

        # Stage A: Only cross-attn, LayerNorms, adapter
        if self.stage == "A":
            param_groups.append({"params": cross_attn_params(self.decoder), "lr": self.lr})
            param_groups.append({"params": layernorm_params(self.decoder), "lr": self.lr})
            param_groups.append({"params": adapter_params(self.decoder), "lr": self.lr})
        # Stage B: Add top decoder blocks with lower LR
        elif self.stage == "B":
            param_groups.append({"params": cross_attn_params(self.decoder), "lr": self.lr})
            param_groups.append({"params": layernorm_params(self.decoder), "lr": self.lr})
            param_groups.append({"params": adapter_params(self.decoder), "lr": self.lr})
            param_groups.append({"params": top_decoder_block_params(self.decoder, self.decoder_top_unfrozen), "lr": self.decoder_top_lr})
        # Stage C: Add encoder with even lower LR
        elif self.stage == "C":
            param_groups.append({"params": cross_attn_params(self.decoder), "lr": self.lr})
            param_groups.append({"params": layernorm_params(self.decoder), "lr": self.lr})
            param_groups.append({"params": adapter_params(self.decoder), "lr": self.lr})
            param_groups.append({"params": top_decoder_block_params(self.decoder, self.decoder_top_unfrozen), "lr": self.decoder_top_lr})
            param_groups.append({"params": [p for p in self.encoder.parameters() if p.requires_grad], "lr": self.encoder_lr})
        else:
            # Default: all params, single LR
            param_groups.append({"params": [p for p in self.parameters() if p.requires_grad], "lr": self.lr})

        optimizer = optim.AdamW(param_groups)
        return optimizer

    # --------------------------------------------------
    def forward(self, src_ids, tgt_ids):
        enc_mask = src_ids.eq(self.pad_id)
        enc_hidden = self.encoder(src_ids, mask=enc_mask, return_hiddens=True)
        dec_input = _shift_right(tgt_ids, self.pad_id, self.bos_id)
        print(enc_hidden.shape)
        print(dec_input.shape)
        logits = self.decoder(dec_input, context=enc_hidden)
        return logits

    # --------------------------------------------------
    def _compute_loss(self, logits, labels):
        """Compute cross entropy loss, ignoring padding tokens."""
        # Shift labels to align with predictions
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
    
    def training_step(self, batch, batch_idx):
        src = batch["encoder_ids"] if "encoder_ids" in batch else batch["input_ids"]
        tgt = batch["decoder_ids"] if "decoder_ids" in batch else batch["input_ids"]
        
        logits = self(src, tgt)
        loss = self._compute_loss(logits, tgt)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src = batch["encoder_ids"] if "encoder_ids" in batch else batch["input_ids"]
        tgt = batch["decoder_ids"] if "decoder_ids" in batch else batch["input_ids"]
        
        logits = self(src, tgt)
        loss = self._compute_loss(logits, tgt)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
        
    # --------------------------------------------------
    def generate(self, src_ids, prompt_ids=None, max_length=4096, temperature=1.0, top_k=0, top_p=0.9, output_path=None):
        """
        Generate a sequence from a prompt using autoregressive sampling with encoder context.
        
        Args:
            src_ids: Tensor of shape [batch_size, seq_len] containing source token ids for the encoder
            prompt_ids: Optional tensor of shape [batch_size, seq_len] containing prompt token ids for the decoder
                        If None, will start with just BOS token
            max_length: Maximum length of the generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep (0 = disabled)
            top_p: Cumulative probability threshold for nucleus sampling
            output_path: Optional path to save the generated MIDI file
            
        Returns:
            Generated token ids of shape [batch_size, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_size = src_ids.shape[0]
        
        # Move inputs to the correct device
        src_ids = src_ids.to(device)
        
        # Get encoder context
        enc_mask = src_ids.eq(self.pad_id)
        enc_context = self.encoder(src_ids, src_key_padding_mask=enc_mask, return_hidden_states=True)
        
        # Initialize prompt or start with BOS token
        if prompt_ids is None:
            prompt_ids = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        else:
            prompt_ids = prompt_ids.to(device)
        
        # Start with the prompt
        generated = prompt_ids.clone()
        
        # Generate tokens one by one
        with torch.no_grad():
            # Use tqdm for progress tracking during generation
            for _ in tqdm(range(max_length - prompt_ids.shape[1]), desc="Generating tokens"):
                # Get model predictions
                logits = self.decoder(generated, context=enc_context, src_key_padding_mask=enc_mask)
                
                # Only consider the last token's predictions
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the sampled token to the sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Stop if we've generated an EOS token
                if (next_token == self.eos_id).any():
                    # Find the first EOS token in each sequence
                    eos_positions = (generated == self.eos_id).nonzero()
                    if len(eos_positions) > 0:
                        # Group by batch index
                        batch_indices = {}
                        for batch_idx, seq_pos in eos_positions:
                            if batch_idx.item() not in batch_indices or seq_pos < batch_indices[batch_idx.item()]:
                                batch_indices[batch_idx.item()] = seq_pos
                        
                        # If all sequences have an EOS token, stop generation
                        if len(batch_indices) == batch_size:
                            break
        
        if batch_size == 1:
            print(self.tokenizer.decode(generated[0].cpu().tolist()))
            # Save to MIDI file if output path is provided
            if output_path is not None:
                self.tokenizer.ids_to_file(generated[0].cpu().tolist(), output_path)
            
        return generated
        
    # --------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        tokenizer,
        encoder_ckpt: str,
        decoder_ckpt: str = None,
        stage: str = "A",
        decoder_top_unfrozen: int = 4,
        encoder_lr: float = 1e-6,
        decoder_top_lr: float = 5e-6,
        **kwargs,
    ):
        model = cls(
            tokenizer,
            stage=stage,
            decoder_top_unfrozen=decoder_top_unfrozen,
            encoder_lr=encoder_lr,
            decoder_top_lr=decoder_top_lr,
            **kwargs
        )
        
        # Load encoder from checkpoint
        if encoder_ckpt is None:
            print("Creating new encoder")
            model.encoder = ContrastiveStructureEncoder(tokenizer, dim=1536, depth=6, heads=8, max_len=4096, temp=0.08, lambda_local=0.7, lr=1e-5)
        else:
            print("Loading encoder from checkpoint")
            enc = ContrastiveStructureEncoder.load_from_checkpoint(encoder_ckpt, tokenizer=tokenizer)
            model.encoder = enc.model
        
        # Load or create decoder
        if decoder_ckpt is None:
            print("Creating new decoder")
            # Create a new DecoderLM with default or provided kwargs
            dec = DecoderLM(tokenizer=tokenizer, cross_attend=True, **kwargs)
            model.decoder = dec.model
        else:
            print("Loading decoder from checkpoint")
            dec = DecoderLM.load_from_checkpoint(decoder_ckpt, tokenizer=tokenizer, cross_attend=True, strict=False)
            model.decoder = dec.model
        
        # Apply freezing based on stage
        model._apply_stage_freezing()
        
        return model


if __name__ == "__main__":

    from src.MidiDataModule import DataCfg, MidiDataModule
    tokenizer = MusicTokenizerWithStyle()
    data_cfg = DataCfg(
        data_dir   = "cache/real_10-tiny_max4096_limitNone_d352930e.pkl",
        mode       = "real",
        task       = "generative",
        max_len    = 4096,
        seq_limit  = 1000,
        shuffle    = True,
        skip_long  = True,
        val_split  = 0.1
    )
    dm = MidiDataModule(
        cfg         = data_cfg,
        tokenizer   = tokenizer,
        batch_size  = 1,
        num_workers = 4,
        drop_last   = False,
    )
    dm.setup()
    
