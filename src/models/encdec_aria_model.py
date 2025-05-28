# generator_finetuner.py
import json
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from torch.optim import AdamW
from tqdm import tqdm
from aria_generative.model import ModelConfig
from aria_generative.model import TransformerLM
from aria_generative.config import load_model_config
from aria_generative.utils import _load_weight

from SnippetModule import SnippetEncoderLit
from tokenizer import MusicTokenizerWithStyle
"""
Lightning module that fine‑tunes Aria with snippet‑conditioned cross‑attention.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW

from aria_generative.config import load_model_config
from aria_generative.utils import _load_weight
from aria_generative.cross_transformer import ModelConfig, TransformerLM   # <-- NEW import
from SnippetModule import SnippetEncoderLit
from tokenizer import MusicTokenizerWithStyle
class EncDecAriaModel(pl.LightningModule):
    def __init__(
        self,
        generative_config_path: str,
        generative_weights_path: str,
        encoder_ckpt: str,
        lr: float = 1e-5,
        freeze_stage: int = 1,
        thaw_top_k: int = 4,
    ):
        """
        freeze_stage:
          1 — freeze encoder + decoder self-attn/FF; train only cross-attn & LayerNorm
          2 — additionally unfreeze top thaw_top_k decoder blocks
          3 — unfreeze snippet encoder too (small LR)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["lr"])
        self.lr = lr
        self.freeze_stage = freeze_stage
        self.thaw_top_k = thaw_top_k

        # tokenizer & loss
        self.tok = MusicTokenizerWithStyle()
        self.pad_id = self.tok.pad_id
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id)

        # snippet encoder (frozen initially)
        lit = SnippetEncoderLit.load_from_checkpoint(encoder_ckpt, map_location="cpu")
        self.encoder = lit.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # decoder
        cfg = ModelConfig(**load_model_config(generative_config_path), vocab_size=self.tok.vocab_size)
        self.decoder = TransformerLM(cfg)
        if generative_weights_path:
            # Load the checkpoint and print its keys
            state = _load_weight(generative_weights_path, device="cpu")
            print("Checkpoint keys:")
            for key in list(state.keys())[:10]:  # Print first 10 keys to avoid clutter
                print(f"  {key}")
            print(f"  ... ({len(state)} keys total)")
            
            # Print model keys
            print("\nModel keys:")
            model_keys = [name for name, _ in self.decoder.named_parameters()]
            for key in model_keys[:10]:  # Print first 10 keys
                print(f"  {key}")
            print(f"  ... ({len(model_keys)} keys total)")
            
            # Check for key mismatches
            model_keys_set = set(model_keys)
            checkpoint_keys_set = set(state.keys())
            missing_in_model = checkpoint_keys_set - model_keys_set
            missing_in_checkpoint = model_keys_set - checkpoint_keys_set
            
            if missing_in_model:
                print("\nKeys in checkpoint but not in model:")
                for key in list(missing_in_model)[:5]:  # Print first 5
                    print(f"  {key}")
                if len(missing_in_model) > 5:
                    print(f"  ... ({len(missing_in_model)} keys total)")
            
            if missing_in_checkpoint:
                print("\nKeys in model but not in checkpoint:")
                for key in list(missing_in_checkpoint)[:5]:  # Print first 5
                    print(f"  {key}")
                if len(missing_in_checkpoint) > 5:
                    print(f"  ... ({len(missing_in_checkpoint)} keys total)")
            
            # Load the weights
            self.decoder.load_state_dict(state, strict=False)

        # apply initial freezing
        self._apply_freeze(stage=freeze_stage)

    def _apply_freeze(self, stage: int):
        # Stage 1: freeze all decoder except cross-attn & LayerNorm
        for name, param in self.decoder.named_parameters():
            if "proj_cross" in name or "q_cross" in name or "kv_cross" in name or "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if stage >= 2:
            # Stage 2: unfreeze top K decoder blocks (self-attn+FFN)
            blocks = list(self.decoder.decoder.blocks)
            for blk in blocks[-self.thaw_top_k:]:
                for param in blk.parameters():
                    param.requires_grad = True

        if stage >= 3:
            # Stage 3: unfreeze snippet encoder (with small learning rate)
            for param in self.encoder.parameters():
                param.requires_grad = True

    # --------------------------------------------------------------------- #
    def forward(self, batch):
        enc_ids, enc_mask = batch["encoder_ids"], batch["encoder_mask"]
        dec_ids            = batch["decoder_ids"]
        print(f"enc_ids.shape: {enc_ids.shape}")
        print(f"enc_mask.shape: {enc_mask.shape}")
        print(f"dec_ids.shape: {dec_ids.shape}")

        enc_out = self.encoder(enc_ids, enc_mask)   # (B,S_enc,D)
        print(enc_out.shape)

        logits  = self.decoder(dec_ids, enc_out, enc_mask)  # (B,S_dec‑1,V)
        return logits, dec_ids[:, 1:]               # shift‑left targets

    # --------------------------------------------------------------------- #
    def training_step(self, batch, _):
        logits, tgt = self(batch)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        logits, tgt = self(batch)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
    @torch.no_grad()
    def generate(
        self,
        dataloader,
        max_length: int = 3800,
        temperature: float = 0.9,
        top_k: int = 10,
        device: str = "cuda",
        num_batches: int = None,
    ):
        """
        Run inference on batches from `dataloader`.
        Returns a list of generated token sequences (list of lists).
        Sampling via top-k + temperature.

        Args:
            num_batches: If provided, limits generation to this many batches.
        """
        self.eval()
        device = device or next(self.parameters()).device
        results = []

        for batch_idx, batch in enumerate(dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            enc_ids = batch["encoder_ids"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            B = enc_ids.size(0)

            # encode once
            enc_out = self.encoder(enc_ids, enc_mask)  # (B, S_enc, D)

            # start with prefix tokens
            prefix_tokens = []
            prefix_tokens.append(self.tok.encode([('prefix', 'instrument', 'piano')])[0])
            prefix_tokens.append(self.tok.encode(['<S>'])[0])
            
            # create initial decoder input with prefix tokens
            dec_input = torch.tensor([prefix_tokens] * B, dtype=torch.long, device=device)

            # autoregressive loop
            for _ in tqdm(range(max_length), desc="Generating", leave=False):
                logits = self.decoder(dec_input, enc_out, enc_mask)  # (B, L, V)
                next_logits = logits[:, -1, :] / temperature  # (B, V)

                # top-k filtering
                if top_k > 0:
                    vals, idxs = next_logits.topk(top_k, dim=-1)
                    probs = torch.zeros_like(next_logits).scatter_(-1, idxs, F.softmax(vals, dim=-1))
                else:
                    probs = F.softmax(next_logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
                dec_input = torch.cat([dec_input, next_token], dim=1)

                # check if EOS token is generated for all batches
                eos_mask = (next_token == self.tok.eos_id).all(dim=0)
                if eos_mask.item():  # If all sequences in the batch generated EOS
                    break

            # return the generated sequences (without the prefix tokens)
            results.extend(dec_input[:, len(prefix_tokens):].tolist())

        return results
if __name__ == "__main__":
    model = EncDecAriaModel(
        generative_config_path="base-weights/config.json",
        generative_weights_path="base-weights/TokenizerStyleV2-medium-e75.safetensors",
        encoder_ckpt="base-weights/synthetic-encoder-4096-90k/enc_bs32_lr3e-05_seq4096_node1917_20250508_003650/enc-epoch=01-step=2252-val_nt_xent=0.000.ckpt",
        freeze_stage=1,
        thaw_top_k=4
    )
    print(model)
    
    # Test with a sample from SyntheticHierarchicalDataset
    import logging
    import torch
    from tokenizer import MusicTokenizerWithStyle
    from HeiarchicalDataModule import DataCfg, AugmentCfg, MidiDataModule

    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s] %(levelname)s: %(message)s")
    
    # Set device to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = MusicTokenizerWithStyle()
    
    # Create data config
    data_cfg = DataCfg(
        data_dir="datasets/gen1-5k-real",  # Adjust path if needed
        mode="synthetic",
        task="generative",
        max_len=4096,
        num_workers=128,
        val_split=0.1,
        cache_dir="./cache",
        aug=AugmentCfg(enable=False)  # Disable augmentation for testing
    )
    
    # Initialize data module
    dm = MidiDataModule(
        cfg=data_cfg,
        tokenizer=tokenizer,
        batch_size=2,  # Small batch for testing
        num_workers=32,
        drop_last=False
    )
    
    # Setup data
    dm.prepare_data()
    dm.setup()
    
    # Get a batch
    batch = next(iter(dm.val_dataloader()))
    
    # Move batch to CUDA
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Run forward pass
    with torch.no_grad():
        logits, targets = model(batch)
        loss = model.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    
    print(f"Test batch shapes:")
    print(f"  Encoder IDs: {batch['encoder_ids'].shape}")
    print(f"  Encoder IDs: {tokenizer.decode(batch['encoder_ids'][0].tolist())}")
    print(f"  Encoder mask: {batch['encoder_mask'].shape}")
    print(f"  Decoder IDs: {batch['decoder_ids'].shape}")
    print(f"  Decoder mask: {batch['decoder_mask'].shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    print("\nTesting generation...")
    
    # Use a small subset of the validation data for generation
    val_loader = dm.val_dataloader()
    
    # Generate sequences
    generated_sequences = model.generate(
        dataloader=val_loader,
        max_length=1024,  # Shorter for testing
        temperature=0.95,
        top_k=3,
        device=device,  # Explicitly pass CUDA device
        num_batches=1  # Only use one batch for testing
    )
    
    print(f"Generated {len(generated_sequences)} sequences")
    
    if generated_sequences:
        # Get the first generated sequence
        first_sequence = generated_sequences[0]
        print(f"First sequence length: {len(first_sequence)}")
        
        # Create output directory if it doesn't exist
        import os
        output_dir = "generation_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the MIDI file using the tokenizer's ids_to_file method
        output_path = os.path.join(output_dir, "generated_sample.mid")
        tokenizer.ids_to_file(first_sequence, output_path)
        
        # Decode a preview of the tokens for display
        decoded_preview = tokenizer.decode(first_sequence[:100])
        print(f"Decoded sequence preview: {decoded_preview}...")
