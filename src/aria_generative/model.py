"""Training model implementation with extensive shape asserts."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.utils.checkpoint

from torch import nn as nn
from torch.nn import functional as F

# Try to import sageattn, otherwise set to None
try:
    from sageattention import sageattn
    _has_sageattn = True
except ImportError:
    sageattn = None
    _has_sageattn = False


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    ff_mult: int
    drop_p: float
    max_seq_len: int
    grad_checkpoint: bool
    vocab_size: Optional[int] = None
    class_size: Optional[int] = None
    tag_to_id: Optional[dict] = None
    emb_size: Optional[int] = None

    def set_vocab_size(self, vocab_size: int):
        self.vocab_size = vocab_size


class FusedEncoderBlock(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.drop_p = model_config.drop_p
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.max_seq_len = model_config.max_seq_len
        self.d_model = model_config.d_model

        # Attention
        self.mixed_qkv = nn.Linear(
            in_features=model_config.d_model,
            out_features=3 * model_config.d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )

        # FF Layer
        self.ff_gate_proj = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model * model_config.ff_mult,
            bias=False,
        )
        self.ff_up_proj = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model * model_config.ff_mult,
            bias=False,
        )
        self.ff_down_proj = nn.Linear(
            in_features=model_config.d_model * model_config.ff_mult,
            out_features=model_config.d_model,
            bias=False,
        )

        # Pre layer norms
        self.norm1 = nn.LayerNorm(model_config.d_model)
        self.norm2 = nn.LayerNorm(model_config.d_model)

    # ---------------------------------------------------------------------
    # Forward helpers
    # ---------------------------------------------------------------------

    def _assert_input_shapes(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """Validate the shapes expected by _att_block / forward."""
        assert x.dim() == 3, f"x must be (B, S, D), got {x.shape}"
        B, S, D = x.shape
        assert D == self.d_model, (
            f"Last dim of x should equal d_model ({self.d_model}), got {D}"
        )
        # freqs_cis[(S, d_head/2, 2)]
        assert freqs_cis.dim() == 3, (
            "freqs_cis must be (S, d_head//2, 2), got " f"{freqs_cis.shape}"
        )
        assert freqs_cis.size(0) == S, (
            f"freqs_cis 1st dim ({freqs_cis.size(0)}) != seq_len ({S})"
        )
        assert freqs_cis.size(1) * 2 == self.d_head, (
            "freqs_cis second dim should be d_head//2, got "
            f"{freqs_cis.size(1)} for d_head {self.d_head}"
        )

    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        self._assert_input_shapes(x, freqs_cis)
        x = x + self._att_block(self.norm1(x), freqs_cis)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _att_block(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        # x: (B, S, D)
        batch_size, seq_len, _ = x.shape
        mixed_qkv = self.mixed_qkv(x)
        xq, xk, xv = mixed_qkv.chunk(3, -1)

        # Reshape for rotary embeddings
        xq = xq.reshape(batch_size, seq_len, self.n_heads, self.d_head).contiguous()
        xk = xk.reshape(batch_size, seq_len, self.n_heads, self.d_head).contiguous()
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)

        # Sanity‑check reshapes
        for name, tens in zip(("q", "k", "v"), (xq, xk, xv)):
            assert tens.shape == (
                batch_size,
                seq_len,
                self.n_heads,
                self.d_head,
            ), f"{name} shape incorrect, got {tens.shape}"

        # Apply RoPE
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        # After transpose: (B, H, S, D)
        if _has_sageattn:
            att = sageattn(q=xq, k=xk, v=xv, tensor_layout="HND", is_causal=True)
        else:
            att = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, is_causal=True
            )

        # Output back to (B, S, H, D)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        # Final assert before projection
        assert out.shape == (
            batch_size,
            seq_len,
            self.d_model,
        ), f"att output has wrong shape {out.shape}"

        return self.att_proj_linear(out)

    def _ff_block(self, x: torch.Tensor):
        return self.ff_down_proj(F.silu(self.ff_gate_proj(x)) * self.ff_up_proj(x))


class Transformer(nn.Module):
    """Transformer decoder without a language model head."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.freqs_cis = None

        self.tok_embeddings = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_dim=model_config.d_model,
        )

        self.out_layer_norm = nn.LayerNorm(model_config.d_model)
        self.encode_layers = nn.ModuleList(
            [FusedEncoderBlock(model_config) for _ in range(model_config.n_layers)]
        )

    # ---------------------------------------------------------------
    def _assert_src(self, src: torch.Tensor):
        assert src.dim() == 2, f"src must be (B, S), got {src.shape}"
        assert (
            src.size(1) <= self.model_config.max_seq_len
        ), f"seq_len {src.size(1)} exceeds max_seq_len {self.model_config.max_seq_len}"

    # ---------------------------------------------------------------

    def forward(self, src: torch.Tensor, emb: torch.Tensor | None = None):
        self._assert_src(src)
        hidden_states = self.tok_embeddings(src)

        if emb is not None:
            # emb: (B, d_model)
            assert emb.dim() == 2 and emb.size(0) == src.size(0), (
                f"emb must be (B, D), got {emb.shape} for batch {src.size(0)}"
            )
            emb = emb[:, None, :]
            hidden_states = torch.cat([emb, hidden_states[:, :-1, :]], dim=1)

        # Pre‑compute freqs_cis once per device / dtype
        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis(
                seq_len=self.model_config.max_seq_len,
                n_elem=self.model_config.d_model // self.model_config.n_heads,
                base=500000,
                dtype=hidden_states.dtype,
            ).to(src.device)
        freqs_cis = self.freqs_cis[: src.shape[1]]

        # Main blocks
        if self.model_config.grad_checkpoint:
            for layer in self.encode_layers:

                def create_custom_forward(module):
                    def custom_forward(*args):
                        return module(*args)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    freqs_cis,
                    preserve_rng_state=True,
                    use_reentrant=False,
                )
        else:
            for layer in self.encode_layers:
                hidden_states = layer(hidden_states, freqs_cis=freqs_cis)

        return self.out_layer_norm(hidden_states)


class TransformerLM(nn.Module):
    """Transformer decoder with a language modeling head."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        assert model_config.vocab_size is not None, "vocab_size must be set"

        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.lm_head = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)

    def forward(self, src: torch.Tensor):
        hidden = self.model(src)
        logits = self.lm_head(hidden)
        # final shape assert
        assert logits.shape[:2] == src.shape and logits.size(2) == self.lm_head.out_features, (
            f"LM logits shape mismatch {logits.shape}"
        )
        return logits


class TransformerCL(nn.Module):
    """Transformer decoder with a classification head."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        assert model_config.class_size is not None, "class_size must be set"

        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.class_head = nn.Linear(model_config.d_model, model_config.class_size, bias=False)

    def forward(self, src: torch.Tensor):
        hidden = self.model(src)
        logits = self.class_head(hidden)
        assert logits.shape[:2] == src.shape and logits.size(2) == self.class_head.out_features, (
            f"CL logits shape mismatch {logits.shape}"
        )
        return logits


class TransformerLM_CND(nn.Module):
    """Transformer decoder with a language modeling head and optional conditioning."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        assert model_config.vocab_size is not None, "vocab_size must be set"
        assert model_config.emb_size is not None, "emb_size must be set"

        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.lm_head = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)
        self.embedding_adapter = nn.Linear(model_config.emb_size, model_config.d_model, bias=False)

    def forward(self, src: torch.Tensor, emb: torch.Tensor | None = None):
        if emb is not None:
            assert emb.dim() == 2 and emb.size(0) == src.size(0), (
                f"emb must be (B, emb_size), got {emb.shape}"
            )
            emb = self.embedding_adapter(emb)
            hidden = self.model(src, emb)
            logits = self.lm_head(hidden)
            logits = logits[:, 1:, :]
            assert logits.shape[1] == src.shape[1] and logits.shape[0] == src.shape[0]
            return logits
        else:
            dummy_input = torch.zeros(src.size(0), self.embedding_adapter.in_features, device=src.device)
            dummy_output = self.embedding_adapter(dummy_input)
            dummy_loss = dummy_output.sum() * 0.0
            hidden = self.model(src, None)
            logits = self.lm_head(hidden)
            logits = logits + dummy_loss
            return logits


class TransformerEMB(nn.Module):
    """Transformer decoder with an embedding head.

    Args:
        model_config (ModelConfig): Model configuration settings (emb_size must be defined).
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        assert model_config.emb_size is not None

        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.emb_head = nn.Linear(
            model_config.d_model, model_config.emb_size, bias=False
        )

    def forward(
        self,
        src: torch.Tensor,
    ):
        """Compute output embeddings from the transformer.

        Args:
            src (torch.Tensor): Input tensor of token indices with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output embeddings with shape (batch_size, seq_len, emb_size).
        """

        hidden = self.model(src)
        emb = self.emb_head(hidden)

        return emb


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 500000,
    dtype: torch.dtype = torch.bfloat16,
):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

    return cache.to(dtype=dtype)



@torch.jit.script
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    In-place RoPE. Credits to Katherine Crowson:
    x shape (b_sz, s_len, n_head, d_head).
    cos, sin shape (s_len, d_head // 2).
    """

    d = x.shape[-1] // 2
    cos = freqs_cis[..., 0][None, :, None]
    sin = freqs_cis[..., 1][None, :, None]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    tmp = x1.clone()
    x1.mul_(cos).addcmul_(x2, sin, value=-1)
    x2.mul_(cos).addcmul_(tmp, sin, value=1)
    return x
