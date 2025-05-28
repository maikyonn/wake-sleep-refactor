#!/usr/bin/env python
import torch
import torch.nn.functional as F
from src.StariaTokenizer import MusicTokenizerWithStyle
from src.MidiDataModule import DataCfg, AugmentCfg, MidiDataModule
from src.StariaModule import ContrastiveStructureEncoder

def main():
    # ——— 1. Setup DataModule ———————————————————————————————————————
    data_cfg = DataCfg(
        data_dir   = "datasets/gen0-synth_midi_90k",
        mode       = "synthetic",
        task       = "contrastive",
        max_len    = 4096,
        seq_limit  = 1000,
        shuffle    = True,
        skip_long  = True,
        val_split  = 0.1
    )
    tokenizer = MusicTokenizerWithStyle()
    dm = MidiDataModule(
        cfg         = data_cfg,
        tokenizer   = tokenizer,
        batch_size  = 4,
        num_workers = 4,
        drop_last   = False
    )
    dm.prepare_data()
    dm.setup()

    # ——— 2. Load first 5 batches ————————————————————————————————————————————
    from itertools import islice
    tokenizer = MusicTokenizerWithStyle()

    # ——— 3. Initialize & load encoder ——————————————————————————————————
    encoder = ContrastiveStructureEncoder.load_from_checkpoint("checkpoints/midi_contrastive/epochepoch=15-valval/loss=0.0005.ckpt", tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    encoder.eval()
    for batch_idx, batch in enumerate(islice(dm.train_dataloader(), 5)):
        print(f"\n================ Batch {batch_idx+1} ================")
        x1, x2, mask = batch["x1_local"].to(device), batch["x2_local"].to(device), batch["mask_local"].to(device)

        # ——— 4. Compute pooled embeddings ——————————————————————————————————
        with torch.no_grad():
            z1 = encoder(x1, mask)   # (B, H)
            z2 = encoder(x2, mask)   # (B, H)

        # ——— 5. Positive‐pair similarities ————————————————————————————————
        pos_sims = F.cosine_similarity(z1, z2, dim=-1)  # (B,)
        print("\n=== Positive Pair Cosine Similarities ===")
        for i, sim in enumerate(pos_sims.cpu().tolist()):
            print(f"Snippet {i:>2}:  cos(z1, z2) = {sim:.4f}")

        # ——— 6. Full pairwise matrix & off‐diagonals —————————————————————————
        # We'll compare each pooled embedding z1[i] against all z1[j] (including itself).
        matrix = F.cosine_similarity(
            z1.unsqueeze(1),  # (B,1,H)
            z1.unsqueeze(0),  # (1,B,H)
            dim=-1            # → (B,B)
        ).cpu().numpy()

        print("\n=== Full Pairwise Cosine‐Similarity Matrix (z1 vs z1) ===")
        for row in matrix:
            print("  [" + ", ".join(f"{v:.3f}" for v in row) + "]")

        # Off‐diagonals
        B = matrix.shape[0]
        off_diags = [matrix[i][j] for i in range(B) for j in range(B) if i != j]
        print("\n=== Off-Diagonal Similarities (should be low) ===")
        print("  " + ", ".join(f"{v:.4f}" for v in off_diags))

if __name__ == "__main__":
    main()