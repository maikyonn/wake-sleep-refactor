#!/usr/bin/env python
"""
Train Staria music‑generation model with **Lightning Trainer** and modern callbacks.

Key features added:
- **TQDMProgressBar** for rich per‑epoch progress bars.
- **Stochastic Weight Averaging** (starting at epoch 10).
- **ModelCheckpoint** keeps only the single best model (lowest `val_loss`) and the last checkpoint.
- **DeviceStatsMonitor** logs real‑time GPU/CPU and memory stats to your loggers.

The script no longer uses a manual Fabric training loop—everything is delegated to
`lightning.Trainer`, which automatically invokes the callbacks.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from x_transformers import Decoder, Encoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
import pickle
from src.StariaTokenizer import (
    A_SECTION_TOKEN,
    B_SECTION_TOKEN,
    C_SECTION_TOKEN,
    D_SECTION_TOKEN,
    IGNORE_LABEL_IDX,
    MusicTokenizerWithStyle,
    PROMPT_END_TOKEN,
    PROMPT_START_TOKEN,
)
from src.utils_new import music_style_from_labels

# ──────────────────────────────────────────────────────────────────────────────
# Globals & logging
# ──────────────────────────────────────────────────────────────────────────────
L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_SNIPPET_LEN = 256
_SECTION_TOK = {
    "A": A_SECTION_TOKEN,
    "B": B_SECTION_TOKEN,
    "C": C_SECTION_TOKEN,
    "D": D_SECTION_TOKEN,
}
_PAD_ID = 2  # will be overwritten by tokenizer.pad_id at run‑time


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & Dataloader helpers
# ──────────────────────────────────────────────────────────────────────────────
class OnDemandMidiDataset(Dataset):
    """Lazy‑loading dataset that tokenises each MIDI on‑the‑fly."""

    def __init__(
        self,
        path_pkl_file: str,
        tokenizer: MusicTokenizerWithStyle,
        max_len: int = 4096,
        seq_limit: Optional[int] = None,
        use_snippet: bool = True,
    ) -> None:
        super().__init__()
        self.path_pkl_file = path_pkl_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_snippet = use_snippet
        global _PAD_ID
        _PAD_ID = tokenizer.pad_id

        logger.info("Loading dataset index from %s", path_pkl_file)
        with open(path_pkl_file, "rb") as f:
            loaded = pickle.load(f)
        self.records: List[Dict[str, str]] = loaded["data_records"][: seq_limit or None]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        rec = self.records[idx]
        midi_fp, style_fp = rec["midi_file_path"], rec["style_file_path"]
        if not os.path.exists(midi_fp):
            return None

        tokens = self.tokenizer.tokenize_from_file(midi_fp)
        if not tokens:
            return None

        style_labels = None
        if style_fp and os.path.exists(style_fp):
            with open(style_fp) as f:
                txt = f.read().strip()
            style_labels = list(txt) if txt else None
            if style_labels and len(style_labels) != len(tokens):
                style_labels = None  # invalid ⇒ fall back to AR mode

        # ------------------------------------------------------------------
        if self.use_snippet and style_labels:
            enc_prompt = [PROMPT_START_TOKEN]
            runs: List[tuple[str, int, int]] = []
            s = 0
            for i in range(1, len(style_labels)):
                if style_labels[i] != style_labels[i - 1]:
                    runs.append((style_labels[i - 1], s, i - 1))
                    s = i
            runs.append((style_labels[-1], s, len(style_labels) - 1))

            for style_char in music_style_from_labels(style_labels):
                for lab, start, end in runs:
                    if lab == style_char:
                        enc_prompt.append(_SECTION_TOK[lab])
                        seg_end = min(end + 1, start + _SNIPPET_LEN)
                        enc_prompt.extend(tokens[start:seg_end])
                        break
            enc_prompt.append(PROMPT_END_TOKEN)

            enc_ids = self.tokenizer.encode(enc_prompt)[: self.max_len]
            dec_ids = self.tokenizer.encode(tokens)[: self.max_len]
            if not enc_ids or not dec_ids:
                return None

            return {
                "encoder_ids": torch.tensor(enc_ids, dtype=torch.long),
                "decoder_ids": torch.tensor(dec_ids, dtype=torch.long),
            }

        # ── autoregressive fallback ──
        ids = self.tokenizer.encode(tokens)[: self.max_len]
        if not ids:
            return None
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


# ---------------------------------------------------------------------------
def collate_fn(batch: List[Optional[Dict[str, Any]]]):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    if "encoder_ids" in batch[0]:
        enc = [b["encoder_ids"] for b in batch]
        dec = [b["decoder_ids"] for b in batch]
        return {
            "encoder_ids": torch.nn.utils.rnn.pad_sequence(enc, batch_first=True, padding_value=_PAD_ID),
            "decoder_ids": torch.nn.utils.rnn.pad_sequence(dec, batch_first=True, padding_value=_PAD_ID),
        }
    ids = [b["input_ids"] for b in batch]
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=_PAD_ID),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class LitStaria(L.LightningModule):
    def __init__(
        self,
        tokenizer: MusicTokenizerWithStyle,
        dim: int = 1536,
        enc_depth: int = 6,
        enc_heads: int = 8,
        dec_depth: int = 16,
        dec_heads: int = 24,
        max_seq_len: int = 4096,
        lr: float = 1e-4,
        use_snippet: bool = True,
        attn_flash: bool = True,
        rotary_xpos: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.lr = lr
        self.use_snippet = use_snippet
        pad_id = tokenizer.pad_id
        vocab = tokenizer.vocab_size

        if use_snippet:
            self.encoder = TransformerWrapper(
                num_tokens=vocab,
                max_seq_len=max_seq_len,
                attn_layers=Encoder(dim=dim, depth=enc_depth, heads=enc_heads, attn_flash=attn_flash),
            )
            dec_core = TransformerWrapper(
                num_tokens=vocab,
                max_seq_len=max_seq_len,
                attn_layers=Decoder(
                    dim=dim,
                    depth=dec_depth,
                    heads=dec_heads,
                    cross_attend=True,
                    attn_flash=attn_flash,
                    rotary_xpos=rotary_xpos,
                ),
            )
        else:
            dec_core = TransformerWrapper(
                num_tokens=vocab,
                max_seq_len=max_seq_len,
                attn_layers=Decoder(
                    dim=dim,
                    depth=dec_depth,
                    heads=dec_heads,
                    cross_attend=False,
                    attn_flash=attn_flash,
                    rotary_xpos=rotary_xpos,
                ),
            )
        self.decoder = AutoregressiveWrapper(dec_core, ignore_index=IGNORE_LABEL_IDX, pad_value=pad_id)

    # ------------------------------------------------------------------
    def forward(self, src_ids, tgt_ids):
        if self.use_snippet and src_ids is not None:
            enc_out = self.encoder(src_ids, mask=src_ids.ne(_PAD_ID), return_embeddings=True)
            return self.decoder(tgt_ids, context=enc_out, context_mask=src_ids.ne(_PAD_ID))
        return self.decoder(tgt_ids)

    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        if batch is None:
            return None
        if "encoder_ids" in batch:
            loss = self(batch["encoder_ids"], batch["decoder_ids"])
        else:
            loss = self(None, batch["input_ids"])
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    def validation_step(self, batch, _):
        if batch is None:
            return None
        if "encoder_ids" in batch:
            loss = self(batch["encoder_ids"], batch["decoder_ids"])
        else:
            loss = self(None, batch["input_ids"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=self.lr * 0.1)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
            },
        }
        
    # ------------------------------------------------------------------
    def generate(self, 
                 prompt_ids: Optional[torch.Tensor] = None,
                 context_ids: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 **kwargs):
        """
        Generate sequences using the decoder.
        
        Args:
            prompt_ids: Starting tokens for generation [batch_size, seq_len]
            context_ids: Context tokens for encoder (if use_snippet=True) [batch_size, context_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to decoder.generate()
        
        Returns:
            Generated token sequences [batch_size, generated_len]
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Handle context encoding if using snippet mode
        context = None
        context_mask = None
        if self.use_snippet and context_ids is not None:
            context_ids = context_ids.to(device)
            context_mask = context_ids.ne(_PAD_ID)
            context = self.encoder(context_ids, mask=context_mask, return_embeddings=True)
        
        # Handle prompt initialization
        if prompt_ids is None:
            # Start with BOS token if no prompt given
            prompt_ids = torch.tensor([[self.bos_id]], device=device, dtype=torch.long)
        else:
            prompt_ids = prompt_ids.to(device)
        
        # Generate using the AutoregressiveWrapper's generate method
        with torch.no_grad():
            generated = self.decoder.generate(
                prompt_ids,
                max_new_tokens,
                context=context,
                context_mask=context_mask,
                temperature=temperature,
                **kwargs
            )
        
        return generated


# ──────────────────────────────────────────────────────────────────────────────
# Utility to build dataloaders
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    train_index: str,
    val_index: str,
    tokenizer: MusicTokenizerWithStyle,
    batch_size: int = 2,
    num_workers: int = 20,
    max_len: int = 4096,
    use_snippet: bool = True,
):
    train_ds = OnDemandMidiDataset(train_index, tokenizer, max_len=max_len, use_snippet=use_snippet)
    val_ds = OnDemandMidiDataset(val_index, tokenizer, max_len=max_len, use_snippet=use_snippet)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    return train_dl, val_dl


# ──────────────────────────────────────────────────────────────────────────────
# CLI & Training entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Staria with Lightning + callbacks")
    p.add_argument("--train_index", required=True, help="Path to train .pkl index file")
    p.add_argument("--val_index", required=True, help="Path to val .pkl index file")
    p.add_argument("--checkpoint_dir", default="checkpoints/staria", help="Where to save ckpts")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--num_nodes", type=int, default=1)
    p.add_argument("--use_snippet", action="store_true", default=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Loggers ----------------------------------------------------------------
    tb_logger = TensorBoardLogger("logs")
    wb_logger = WandbLogger(project="staria_trainer")

    # Callbacks --------------------------------------------------------------
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        StochasticWeightAveraging(swa_epoch_start=10, swa_lrs=1e-4),
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
    ]

    # Trainer ---------------------------------------------------------------
    trainer = L.Trainer(
        accelerator="cuda",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision="bf16-mixed",
        max_epochs=args.num_epochs,
        gradient_clip_val=1.0,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        callbacks=callbacks,
        logger=[tb_logger, wb_logger],
    )

    # Data & model -----------------------------------------------------------
    tokenizer = MusicTokenizerWithStyle()
    train_dl, val_dl = make_dataloaders(
        args.train_index,
        args.val_index,
        tokenizer,
        batch_size=args.batch_size,
        use_snippet=args.use_snippet,
    )

    model = LitStaria(tokenizer=tokenizer, use_snippet=args.use_snippet)

    # Fit -------------------------------------------------------------------
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume_from_checkpoint)

    logger.info("Training complete. Best checkpoint: %s", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
