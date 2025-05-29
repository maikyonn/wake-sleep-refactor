# encoder_module.py
# ---------------------------------------------------------------------
# Lightweight snippet encoder for hierarchical music modelling
# ---------------------------------------------------------------------
import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.StariaTokenizer import MusicTokenizerWithStyle, A_SECTION_TOKEN, B_SECTION_TOKEN, C_SECTION_TOKEN, D_SECTION_TOKEN

# ---------------------------------------------------------------------
# 1.  Positional Encoding (same max_len as DataModule default = 4096)
# ---------------------------------------------------------------------
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, hidden_size: int, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.pe = nn.Embedding(max_len, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T)
        returns (B, T, H) added to token embeddings
        """
        # positions: 0,1,2,… but keep 0 for pad tokens
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos = pos.expand_as(x)                       # (B, T)
        pos = pos.masked_fill(x == self.pad_idx, 0)  # pad positions stay 0
        return self.pe(pos)


# ---------------------------------------------------------------------
# 2.  Encoder core
# ---------------------------------------------------------------------
class SnippetEncoder(nn.Module):
    """
    Tiny transformer encoder (6 layers, 8‑head, 512‑d) with optional
    weight‑sharing of the token embedding matrix with a pre‑trained
    Aria LM.

    Args
    ----
    vocab_size:          int  –  tokenizer.vocab_size
    embed_dim:           int  –  default 512
    n_layers:            int  –  default 6
    n_heads:             int  –  default 8
    dropout:             float
    pad_idx:             int   –  tokenizer.pad_id
    pretrained_embeddings: Optional[torch.Tensor]  –  weight to copy into
                        self.tok_emb.weight
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 4096,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        tokenizer = MusicTokenizerWithStyle()
    
        
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == self.tok_emb.weight.shape
            self.tok_emb.weight.data[:] = pretrained_embeddings

        self.pos_emb = LearnedPositionalEncoding(max_len, embed_dim, pad_idx)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,    # (B, T)
        attention_mask: torch.Tensor  # (B, T)  1 = keep, 0 = pad
    ) -> torch.Tensor:
        tok = self.tok_emb(input_ids)         # (B,T,H)
        tok = tok + self.pos_emb(input_ids)
        tok = self.layer_norm(tok)
        key_padding_mask = attention_mask == 0       # True where pad
        h = self.encoder(tok, src_key_padding_mask=key_padding_mask)
        return h                                     # (B,T,H)


# ---------------------------------------------------------------------
# 3.  Lightning wrapper
# ---------------------------------------------------------------------

class SnippetEncoderLit(pl.LightningModule):
    """
    Contrastive (NT‑Xent) pre‑training.  Expects each batch to carry:
        { "x1": (B,T),  "x2": (B,T),  "mask": (B,T) }
    """
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embed_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 4096,
        lr: float = 1e-5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_embeddings"])
        # core encoder
        self.encoder = SnippetEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
            pad_idx=pad_idx,
            pretrained_embeddings=pretrained_embeddings,
        )
        self.temperature = 0.1
        self.pad_idx = pad_idx
        self.lr = lr

    # ---------- helper ----------------------------
    def pooled(self, x, m):
        h = self.encoder(x, m)               # (B,T,H)
        m = m.unsqueeze(-1)
        z = torch.sum(h * m, dim=1) / torch.clamp(m.sum(1), min=1e-6)
        return F.normalize(z, dim=-1)        # (B,H)

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a single sequence (1D or 2D tensor) and return the pooled embedding.
        input_ids: (T,) or (1,T) or (B,T)
        attention_mask: (T,) or (1,T) or (B,T), optional. If None, will be inferred (all ones).
        Returns: (H,) if input is (T,), else (B,H)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # (1,T)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        pooled = self.pooled(input_ids, attention_mask)  # (B,H)
        if pooled.size(0) == 1:
            return pooled[0]
        return pooled

    def training_step(self, batch, batch_idx):
        z1 = self.pooled(batch["x1"], batch["mask"])
        z2 = self.pooled(batch["x2"], batch["mask"])
        loss = self.nt_xent(z1, z2)
        self.log("train/nt_xent", loss, prog_bar=True)
        # Log the current learning rate to wandb (and Lightning logger)
        opt = self.optimizers()
        if not isinstance(opt, list):
            opt = [opt]
        lr = opt[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z1 = self.pooled(batch["x1"], batch["mask"])
        z2 = self.pooled(batch["x2"], batch["mask"])
        loss = self.nt_xent(z1, z2)
        self.log("val/nt_xent", loss, prog_bar=True)

    # ---------- NT‑Xent implementation -------------
    def nt_xent(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)              # (2B,H)
        sim = torch.matmul(z, z.T) / self.temperature
        sim.masked_fill_(torch.eye(2*B, device=z.device, dtype=torch.bool), -1e9)
        pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, pos)

    # ---------- optim ------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9,0.95))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100000)
        # Lightning expects a dict for scheduler if you want to monitor/interval
        scheduler = {
            "scheduler": sched,
            "interval": "step",  # or "epoch"
            "frequency": 1,
            "monitor": None,
        }
        return [opt], [scheduler]