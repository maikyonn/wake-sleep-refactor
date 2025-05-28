#!/usr/bin/env python
# eval_midi_classifier_post.py
# ----------------------------------------------------------
# Evaluate a MidiClassifier on synthetic **or** real test‑sets
# and export:
#   • midi_predictions.csv      (all rows)
#   • high_confidence.csv       (rows with ≥0.90 confidence
#                                *and* predicted_music_style != 'A')
# ----------------------------------------------------------

from __future__ import annotations
import os, re, csv, shutil, logging
from argparse import ArgumentParser
from datetime import datetime
from typing import List, Dict

from tqdm import tqdm
import torch
from torchmetrics import Accuracy

from tokenizer import MusicTokenizerWithStyle, IGNORE_LABEL_IDX
from MidiDataModule import MidiDataModule, DataCfg, AugmentCfg
from MidiClassifierModel import MidiClassifier
from utils_new import (
    filter_significant_styles, extract_style_change_timestamps,
    condense_style_sequence
)

LOGGER = logging.getLogger("eval")


# ──────────────────────────────────────────────────────────
#  BUILD TEST DATAMODULE
# ──────────────────────────────────────────────────────────
def build_test_dm(test_dir: str, mode: str,
                  batch_size: int, max_len: int, num_workers: int,
                  seq_limit: int | None = None):
    cfg = DataCfg(
        data_dir=test_dir,
        mode=mode,
        max_len=max_len,
        cache_dir="./cache",
        num_workers=num_workers,
        shuffle=(mode == "synthetic"),
        aug=AugmentCfg(enable=False),
        seq_limit=seq_limit,
    )
    return MidiDataModule(
        cfg,
        MusicTokenizerWithStyle(),
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=0.0,
        drop_last=False,
    )


# ──────────────────────────────────────────────────────────
#  MAIN EVALUATION LOOP
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: MidiClassifier,
             loader,
             tokenizer: MusicTokenizerWithStyle,
             device: torch.device,
             results_dir: str):

    has_gt = "style_label_indices" in loader.dataset[0]
    overall_acc = None
    if has_gt:
        overall_acc = Accuracy(
            task="multiclass",
            num_classes=len(tokenizer.idx_to_style),
            ignore_index=IGNORE_LABEL_IDX,
        ).to(device)

    all_rows: List[Dict]      = []
    high_rows: List[Dict]     = []
    success = skipped = total_tokens = 0

    midi_dir = os.path.join(results_dir, "midi_files")
    os.makedirs(midi_dir, exist_ok=True)

    model.to(device).eval()

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        paths     = batch["file_path"]

        if has_gt:
            gt_batch = batch["style_label_indices"].to(device)

        logits = model(input_ids, attn_mask.bool())
        preds  = logits.argmax(-1)
        conf   = torch.softmax(logits, -1)                 # (B, L, C)

        for i, path in enumerate(paths):
            try:
                mask     = attn_mask[i].bool()
                ids      = input_ids[i][mask].cpu().tolist()
                pred_idx = preds[i][mask].cpu().tolist()
                conf_seq = conf[i][mask].max(-1).values.cpu().numpy()
                labels   = [tokenizer.idx_to_style[j] for j in pred_idx]

                condensed     = condense_style_sequence(labels)
                significant   = filter_significant_styles(labels)
                sig_condensed = condense_style_sequence(significant)

                timestamps = extract_style_change_timestamps(
                    ids, significant, tokenizer=tokenizer
                )

                # average confidence per style letter
                style_conf: Dict[str, float] = {}
                for lab in set(labels):
                    idxs = [k for k, x in enumerate(labels) if x == lab]
                    style_conf[lab] = float(conf_seq[idxs].mean())
                conf_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(style_conf.items()))

                row = {
                    "file_id": os.path.basename(path),
                    "significant_prediction": sig_condensed,
                    "predicted_music_style": "".join(
                        re.findall(r"([A-D])x\d+", sig_condensed)
                    ),
                    "style_change_timestamps": "; ".join(
                        f"{s}:{t}" for s, t in timestamps if t
                    ),
                    "num_tokens": len(ids),
                    "confidence_scores": conf_str,
                    "prediction": condensed,
                }

                if has_gt:
                    gt_idxs = [j for j in gt_batch[i][mask].cpu().tolist()
                               if j != IGNORE_LABEL_IDX]
                    if gt_idxs:
                        gt_labels = [tokenizer.idx_to_style[j] for j in gt_idxs]
                        row["ground_truth"] = condense_style_sequence(gt_labels)
                        overall_acc.update(
                            torch.tensor(pred_idx, device=device),
                            torch.tensor(gt_idxs, device=device),
                        )

                all_rows.append(row)

                # ---------- high‑confidence filter ----------
                letters = set(row["predicted_music_style"])
                if (letters                                   # not empty
                        and letters != {"A"}                  # ignore pure 'A'
                        and all(style_conf.get(l, 0) >= 0.90 for l in letters)):
                    high_rows.append(row)
                    
                    # copy MIDI only for high confidence predictions
                    if os.path.isfile(path):
                        shutil.copy2(path, os.path.join(midi_dir, os.path.basename(path)))

                success      += 1
                total_tokens += len(ids)

            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Error on %s – %s", path, exc)
                skipped += 1

    # ────────────────────────── summary + files ──────────────────────────
    LOGGER.info("Processed %d files (skipped %d) – %d tokens",
                success, skipped, total_tokens)
    if overall_acc:
        LOGGER.info("Overall token accuracy: %.3f", overall_acc.compute().item())
    LOGGER.info("High‑confidence rows kept: %d", len(high_rows))

    # summary text ---------------------------------------------------------
    with open(os.path.join(results_dir, "processing_summary.txt"), "w") as f:
        f.write("===== PROCESSING SUMMARY =====\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Total files processed: {success}\n")
        f.write(f"Total files skipped:  {skipped}\n")
        f.write(f"Total tokens handled: {total_tokens}\n")
        if overall_acc:
            f.write(f"Overall token accuracy: {overall_acc.compute().item():.3f}\n")
        f.write(f"High‑confidence rows: {len(high_rows)}\n")

    # CSV helpers ----------------------------------------------------------
    base_fields = ["file_id"]
    if has_gt:
        base_fields.append("ground_truth")
    base_fields += [
        "significant_prediction", "predicted_music_style",
        "style_change_timestamps", "num_tokens",
        "confidence_scores", "prediction",
    ]

    def write_csv(name: str, rows: List[Dict]):
        out_path = os.path.join(results_dir, name)
        with open(out_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=base_fields)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("%s written (%d rows)", out_path, len(rows))

    write_csv("midi_predictions.csv", all_rows)
    write_csv("high_confidence.csv",  high_rows)
    LOGGER.info("MIDI copies saved to %s", midi_dir)


# ──────────────────────────────────────────────────────────
#  ENTRY‑POINT
# ──────────────────────────────────────────────────────────
def main():
    p = ArgumentParser(description="Evaluate MidiClassifier and export CSVs (incl. high‑confidence).")
    p.add_argument("--test_dir",    required=True)
    p.add_argument("--ckpt_path",   required=True)
    p.add_argument("--mode",       choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--max_length",  type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=128)
    p.add_argument("--results_dir", default="./test_results")
    p.add_argument("--seq_limit",   type=int, default=None,
                   help="Optionally limit number of sequences processed")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    tok = MusicTokenizerWithStyle()
    model = MidiClassifier.load_from_checkpoint(
        args.ckpt_path,
        vocab_size=tok.vocab_size,
        n_classes=len(tok.idx_to_style),
        pad_id=tok.pad_id,
    )

    dm = build_test_dm(
        args.test_dir, args.mode,
        args.batch_size, args.max_length,
        args.num_workers, args.seq_limit,
    )
    dm.setup(stage="test")
    loader = dm.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)
    evaluate(model, loader, tok, device, args.results_dir)


if __name__ == "__main__":
    main()
