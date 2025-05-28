"""simple_transformer.py
========================
A **self‑contained** PyTorch re‑implementation of the minimal subset of
`x_transformers` you used before.  Drop this file into your project,
`import simple_transformer as st`, and carry on – no external deps.

Key design goals
----------------
* Familiar *wrapper* API – just replace `x_transformers` import with this
  module.
* **Toggleable cross‑attention** in the decoder so you can pre‑train it on
  next‑token prediction *without* an encoder, then enable cross‑attention for
  fine‑tuning.
* Causal masking, padding masking, learned positional embeddings.
* Ships with an example at the bottom (runnable: `python simple_transformer.py`).

Usage snippet
-------------
```python
from simple_transformer import Encoder, Decoder, TransformerWrapper

encoder = TransformerWrapper(
    num_tokens=20_000,
    max_seq_len=1024,
    attn_layers=Encoder(dim=512, depth=12, heads=8)
).cuda()

# «pre‑training phase» – no cross‑attention yet
decoder = TransformerWrapper(
    num_tokens=20_000,
    max_seq_len=1024,
    attn_layers=Decoder(dim=512, depth=6, heads=8, cross_attend=False)
).cuda()

# … later, enable cross‑attention and pass encoder context:
decoder.attn_layers.enable_cross_attention()
logits = decoder(input_ids, context=encoder(src_ids))
```
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Upper‑triangular causal mask (1 for *keep*, 0 for *masked*)."""
    return torch.tril(torch.ones((sz, sz), dtype=torch.bool, device=device))

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings (shape: [1, max_len, dim])."""
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pos_emb(positions)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # Self‑attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # Feed‑forward
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, cross_attend: bool = True, dropout: float = 0.1):
        super().__init__()
        self.cross_attend = cross_attend
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        if cross_attend:
            self.cross_attn_layer = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.norm_cross = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    # --------------------------------------------------
    def enable_cross_attention(self, enabled: bool = True):
        if enabled and not self.cross_attend:
            raise ValueError("Cross‑attention was disabled at construction time – you need to rebuild layer with cross_attend=True.")
        self.cross_attend = enabled

    # --------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # causal self‑attention ------------------------------------------------
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # optional cross‑attention -------------------------------------------
        if self.cross_attend and context is not None:
            cross_out, _ = self.cross_attn_layer(
                x, context, context,
                key_padding_mask=ctx_key_padding_mask,
                need_weights=False,
            )
            x = x + self.dropout(cross_out)
            x = self.norm_cross(x)

        # feed‑forward --------------------------------------------------------
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        return x


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(dim, heads, dropout) for _ in range(depth)
        ])
        self.is_decoder = False
        self.dim = dim

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, cross_attend: bool = True, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(dim, heads, cross_attend=cross_attend, dropout=dropout) for _ in range(depth)
        ])
        self.is_decoder = True
        self.dim = dim
        self._initial_cross_attend = cross_attend

    def enable_cross_attention(self, enabled: bool = True):
        for layer in self.layers:
            if not isinstance(layer, DecoderLayer):
                continue
            layer.enable_cross_attention(enabled)

    # --------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        seq_len = x.size(1)
        causal_mask = (~subsequent_mask(seq_len, x.device)).to(x.dtype)  # MultiheadAttention expects float mask – bool works on >=2.1
        for layer in self.layers:
            x = layer(
                x,
                context=context,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                ctx_key_padding_mask=ctx_key_padding_mask,
            )
        return x

# ---------------------------------------------------------------------------
# TransformerWrapper (token + pos + stack + logits)
# ---------------------------------------------------------------------------
class TransformerWrapper(nn.Module):
    def __init__(self, num_tokens: int, max_seq_len: int, attn_layers: nn.Module, dropout: float = 0.1):
        super().__init__()
        dim = attn_layers.dim
        self.token_emb = TokenEmbedding(num_tokens, dim)
        self.pos_emb = PositionalEmbedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_layers = attn_layers
        self.to_logits = nn.Linear(dim, num_tokens)
        self.max_seq_len = max_seq_len

    # --------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,  # <<< ADD THIS ARGUMENT
    ) -> torch.Tensor:
        """Returns logits of shape [B, T, vocab] or hidden states [B, T, dim]."""
        b, t = input_ids.shape
        assert t <= self.max_seq_len, "Sequence length exceeds model limit"

        positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        if self.attn_layers.is_decoder:
            x = self.attn_layers(
                x,
                context=context,
                tgt_key_padding_mask=tgt_key_padding_mask,
                ctx_key_padding_mask=src_key_padding_mask,
            )
        else: # Encoder path
            x = self.attn_layers(x, key_padding_mask=src_key_padding_mask)

        if return_hidden_states:
            return x

        logits = self.to_logits(x)
        return logits


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vocab = 100
    maxlen = 32
    batch = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = TransformerWrapper(vocab, maxlen, Encoder(dim=64, depth=2, heads=4)).to(device)
    dec = TransformerWrapper(vocab, maxlen, Decoder(dim=64, depth=2, heads=4, cross_attend=False)).to(device)

    x = torch.randint(0, vocab, (batch, maxlen), device=device)
    logits_pretrain = dec(x)  # no encoder context yet
    print("Decoder pre‑train logits:", logits_pretrain.shape)

    # fine‑tune with cross‑attention enabled --------------------------------
    dec.attn_layers.enable_cross_attention(True)
    enc_out = enc(x).detach()  # pretend encoder is frozen
    logits_ft = dec(x, context=enc_out)
    print("Decoder fine‑tune logits:", logits_ft.shape)

