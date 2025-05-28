"""
utils.py  ·  helper functions for MIDI structure analysis
---------------------------------------------------------
All functions below are *pure* (no side‑effects or global state) except
`log_music_style`, which only appends one line to a local text file so it
remains safe in multi‑process dataloaders.
"""

from __future__ import annotations

import math
import re
from itertools import groupby
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Condensing, filtering, and relabelling style sequences
# --------------------------------------------------------------------------- #

def condense_style_sequence(labels: Sequence[str]) -> str:
    """
    Collapse a sequence like ['A','A','B'] → 'Ax2, Bx1'.

    Runs of identical labels are counted; the order is preserved and
    comma‑separated. Returns an empty string for empty input.
    """
    if not labels:
        return ""

    parts: list[str] = [
        f"{label}x{sum(1 for _ in group)}"
        for label, group in groupby(labels)
    ]
    return ", ".join(parts)


def filter_significant_styles(
    labels: Sequence[str],
    min_prop: float = 0.05,
) -> List[str]:
    """
    1. Split the sequence into contiguous runs.
    2. Drop every run whose length < `min_prop` · N.
    3. Expand the remaining runs back to the original total length
       while preserving their relative proportions.

    Finally, convert the sequence to **relative labels**
    (first unique run → 'A', next new run → 'B', etc.).

    Example
    -------
    ['A','A','A','B','B','A']  with min_prop=.15 (N=6)
        → keep 'Ax3' (50 %), drop 'Bx2' (33 %), keep 'Ax1' (17 %)
        → remaining runs = [A×3, A×1] = 4 tokens
        → rescale to 6 tokens → [A×4, A×2]
        → relative labelling → ['A','A','A','A','A','A']
    """
    if len(labels) <= 1:
        return list(labels)

    N          = len(labels)
    threshold  = math.ceil(min_prop * N)

    runs: list[Tuple[str, int]] = [(label, sum(1 for _ in grp))
                                   for label, grp in groupby(labels)]

    # keep only “large” runs
    kept_runs = [(lab, ln) for lab, ln in runs if ln >= threshold]
    if not kept_runs:                                  # nothing survives ⇒ empty
        return []

    # length after dropping
    kept_len = sum(ln for _, ln in kept_runs)
    if kept_len == N:                                  # nothing to rescale
        expanded = [lab for lab, ln in kept_runs for _ in range(ln)]
    else:
        # scale each run length up so that new_total == N
        scale = N / kept_len
        new_lengths = [max(1, round(ln * scale)) for _, ln in kept_runs]

        # rounding may overshoot/undershoot by 1–2 tokens → correct
        diff = N - sum(new_lengths)
        for i in range(abs(diff)):
            j = i % len(new_lengths)
            new_lengths[j] += 1 if diff > 0 else -1

        expanded = [lab for (lab, _), ln_new in zip(kept_runs, new_lengths)
                     for _ in range(ln_new)]

    return relative_label_sequence(expanded)


def relative_label_sequence(labels: Iterable[str]) -> List[str]:
    """
    Map first‑seen label → 'A', next unseen → 'B', etc.  
    Provides a canonical “ABA…” view of any alphabet.

    Example: ['X','Y','Z','X'] → ['A','B','C','A']
    """
    mapping: dict[str, str] = {}
    next_ord = ord("A")
    rel: list[str] = []

    for lab in labels:
        if lab not in mapping:
            mapping[lab] = chr(next_ord)
            next_ord += 1
        rel.append(mapping[lab])
    return rel


def get_music_style_from_condensed(condensed: str) -> str:
    """
    Strip the 'x<count>' parts from a condensed string.

    'Ax3, Bx2, Ax1' → 'ABA'  
    'Ax4, Bx4, Cx4' → 'ABC'
    """
    return "".join(re.findall(r"([A-D])x\d+", condensed))

def music_style_from_labels(
    labels: Sequence[str],
    min_prop: float = 0.05,
) -> str:
    """
    Process a sequence of style labels to extract the overall music style.
    
    This function combines filtering significant styles and converting to a
    condensed music style representation.
    
    Parameters
    ----------
    labels : Sequence[str]
        The sequence of style labels to process
    min_prop : float, default=0.05
        Minimum proportion threshold for significant runs
        
    Returns
    -------
    str
        A string representing the music style (e.g., 'ABA', 'ABACA')
        
    Example
    -------
    ['X','X','X','Y','Y','X'] with min_prop=0.15
        → filter significant styles → ['A','A','A','B','B','A']
        → get music style → 'ABA'
    """
    if not labels:
        return ""
        
    # Filter out insignificant style runs and convert to relative labeling
    filtered_labels = filter_significant_styles(labels, min_prop=min_prop)
    
    # Convert to condensed representation
    condensed = condense_style_sequence(filtered_labels)
    
    # Extract the music style from the condensed representation
    return get_music_style_from_condensed(condensed)


# --------------------------------------------------------------------------- #
#  Timestamps of style changes
# --------------------------------------------------------------------------- #

def extract_style_change_timestamps(
    midi_tokens: Sequence[int],
    style_labels: Sequence[str],
    *,
    tokenizer,
    min_prop: float = 0.05,
) -> List[Tuple[str, str | None]]:
    """
    Return a list of `(style_letter, timestamp)` pairs at *major* style changes
    (i.e. the borders between significant runs).

    • Small runs (< `min_prop` of the full piece) are ignored.  
    • `timestamp` is a mm:ss.sss string; `None` if decoding fails.

    The function assumes the tokenizer implements:

        decode(token_ids)  -> bytes | str
        calc_length_ms(decoded, onset=False) -> int
    """
    if len(style_labels) <= 1:
        return []

    N         = len(style_labels)
    threshold = math.ceil(min_prop * N)

    # contiguous runs and their start indices
    runs: list[Tuple[str, int, int]] = []          # label, start, end (incl.)
    start = 0
    for i in range(1, N):
        if style_labels[i] != style_labels[i - 1]:
            runs.append((style_labels[i - 1], start, i - 1))
            start = i
    runs.append((style_labels[-1], start, N - 1))

    # keep only significant runs
    significant = [run for run in runs if run[2] - run[1] + 1 >= threshold]
    if len(significant) <= 1:
        return []

    change_indices = [seg[1] for seg in significant[1:]]  # where run i starts

    out: list[Tuple[str, str | None]] = []
    for idx in change_indices:
        try:
            ts_ms = tokenizer.calc_length_ms(
                tokenizer.decode(midi_tokens[: idx + 1]), onset=False
            )
            mm, ss = divmod(ts_ms / 1000, 60)
            timestamp = f"{int(mm)}:{ss:06.3f}" if mm else f"{ss:.3f}s"
        except Exception:                # if tokenizer.decode fails
            timestamp = None
        out.append((style_labels[idx], timestamp))
    return out


# --------------------------------------------------------------------------- #
#  Other helpers
# --------------------------------------------------------------------------- #

def get_midi_ids_from_style_ids(
    midi_ids: Sequence[int],
    style_ids: Sequence[int],
) -> List[int]:
    """Return the subset of `midi_ids` whose indices appear in `style_ids`."""
    return [midi_ids[i] for i in style_ids]


def log_music_style(style_str: str, logfile: str = "music_styles_log.txt") -> None:
    """
    Append one condensed style string per line.  A no‑op for empty input.
    Designed to be *safe* when called from many workers: append‑only, no locks.
    """
    if style_str:
        try:
            Path(logfile).expanduser().resolve().write_text(
                style_str + "\n", encoding="utf-8", append=True  # `pathlib` ≥3.12
            )
        except Exception:  # pragma: no cover – best‑effort logging
            pass
