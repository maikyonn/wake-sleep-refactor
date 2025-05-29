import os
import random
import hashlib
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from src.StariaTokenizer import (
    STYLE_LABEL_MAP, IGNORE_LABEL_IDX, MusicTokenizerWithStyle, ID_TO_STYLE_MAP,
    PROMPT_START_TOKEN, PROMPT_END_TOKEN, A_SECTION_TOKEN, B_SECTION_TOKEN, 
    C_SECTION_TOKEN, D_SECTION_TOKEN
)

def get_prompt_from_midi_snippets(
    snippets: List[List[str]], 
    music_style: str, 
    tokenizer: "MusicTokenizerWithStyle", 
    max_prompt_length: int = 256
):
    """
    Construct a prompt in the format:
    <PROMPT_START> <music_style_token> <A_SECTION> (snippet) <B_SECTION> (snippet) ... <PROMPT_END>
    Each snippet corresponds to a style section in order: [A_snippet, B_snippet, ...].
    The music_style string determines the order of sections in the prompt.

    Args:
        snippets: List of lists of token strings, e.g. [A_snippet, B_snippet, ...]
        music_style: String like "ABA" or "ABAC"
        tokenizer: MusicTokenizerWithStyle instance
        max_prompt_length: Maximum total number of tokens in the prompt (including all indicators and snippets)

    Returns:
        List of prompt tokens (strings, including indicator tokens)
    """
    # Map style characters to snippet indices
    style_to_snippet_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    # Compose the structure form token, e.g. "<ABA>"
    structure_form_token = f"<{music_style}>"
    # Use the correct attribute name for structure form tokens
    if hasattr(tokenizer, "STRUCTURE_FORMS"):
        known_forms = set(f"<{form}>" for form in tokenizer.STRUCTURE_FORMS)
        if structure_form_token not in known_forms:
            # If not in known forms, just use the string anyway
            pass

    # Calculate the number of style sections
    num_style_sections = len(music_style)
    # Number of special tokens: <PROMPT_START>, <music_style_token>, one <X_SECTION> per section, <PROMPT_END>
    num_special_tokens = 2 + num_style_sections + 1

    # Calculate available space for snippets
    available_for_snippets = max_prompt_length - num_special_tokens
    snippet_length = max(0, available_for_snippets // num_style_sections) if num_style_sections > 0 else 0

    prompt_tokens = [PROMPT_START_TOKEN, structure_form_token]

    for style_char in music_style:
        # Add the style indicator
        if style_char == 'A':
            style_indicator = A_SECTION_TOKEN
        elif style_char == 'B':
            style_indicator = B_SECTION_TOKEN
        elif style_char == 'C':
            style_indicator = C_SECTION_TOKEN
        elif style_char == 'D':
            style_indicator = D_SECTION_TOKEN
        else:
            style_indicator = None

        prompt_tokens.append(style_indicator)

        # Add tokens for this style section
        snippet_idx = style_to_snippet_idx.get(style_char, None)
        if snippet_idx is not None and snippet_idx < len(snippets):
            snippet = snippets[snippet_idx]
            tokens_to_add = snippet[:snippet_length]
            prompt_tokens.extend(tokens_to_add)
            # Pad if not enough tokens in snippet
            if len(tokens_to_add) < snippet_length:
                pad_amt = snippet_length - len(tokens_to_add)
                pad_token = tokenizer.decode([tokenizer.pad_id])[0]
                prompt_tokens.extend([pad_token] * pad_amt)
        else:
            # No snippet for this style, pad
            pad_token = tokenizer.decode([tokenizer.pad_id])[0]
            prompt_tokens.extend([pad_token] * snippet_length)

    # Add the end indicator
    prompt_tokens.append(PROMPT_END_TOKEN)

    # Truncate if over max_prompt_length (shouldn't happen, but just in case)
    if len(prompt_tokens) > max_prompt_length:
        prompt_tokens = prompt_tokens[:max_prompt_length-1] + [PROMPT_END_TOKEN]

    return prompt_tokens

def get_prompt_from_midi_style_ids(
    input_tokens, 
    style_ids, 
    tokenizer, 
    max_prompt_length=512
):
    """
    Construct a prompt in the format:
    <PROMPT_START> <music_style_token> <A_SECTION> (snippet) <B_SECTION> (snippet) ... <PROMPT_END>
    from MIDI tokens and style IDs.

    Args:
        input_tokens: Array of input token IDs
        style_ids: Array of style IDs
        tokenizer: MusicTokenizerWithStyle instance
        max_prompt_length: Maximum total number of tokens in the prompt (including all indicators and snippets)

    Returns:
        Tuple of (prompt_token_ids, prompt_token_strings)
    """
    # Convert style IDs to style tokens
    input_tokens = input_tokens.tolist()
    style_ids = style_ids.tolist()
    style_tokens = [tokenizer.idx_to_style[idx] for idx in style_ids]
    
    # Filter significant styles and get music style
    style_seqs = filter_significant_styles(style_tokens)
    music_style = get_music_style_from_condensed_style_sequence(style_seqs)
    structure_form_token = f"<{music_style}>"

    # Split input_tokens based on style transitions
    style_token_groups = {}
    current_style = style_seqs[0]
    start_idx = 0
    
    for i in range(len(style_seqs)):
        if i == len(style_seqs) - 1 or style_seqs[i+1] != current_style:
            # End of a group
            style_idx = tokenizer.style_to_idx[current_style]
            if style_idx not in style_token_groups:
                style_token_groups[style_idx] = []
            style_token_groups[style_idx].append(input_tokens[start_idx:i+1])
            if i < len(style_seqs) - 1:
                current_style = style_seqs[i+1]
                start_idx = i+1

    num_style_sections = len(music_style)
    if num_style_sections == 0:
        prompt_tokens = [PROMPT_START_TOKEN, PROMPT_END_TOKEN]
        prompt = [tokenizer.encode([PROMPT_START_TOKEN])[0], tokenizer.encode([PROMPT_END_TOKEN])[0]]
        return prompt, prompt_tokens

    # Number of special tokens: <PROMPT_START>, <music_style_token>, one <X_SECTION> per section, <PROMPT_END>
    num_special_tokens = 2 + num_style_sections + 1
    available_for_snippets = max_prompt_length - num_special_tokens
    snippet_length = max(0, available_for_snippets // num_style_sections) if num_style_sections > 0 else 0

    # Construct the prompt
    prompt_tokens = [PROMPT_START_TOKEN, structure_form_token]
    prompt = [
        tokenizer.encode([PROMPT_START_TOKEN])[0],
        tokenizer.encode([structure_form_token])[0]
    ]

    # For each style in the music_style (e.g., ABA)
    for style_char in music_style:
        style_idx = tokenizer.style_to_idx[style_char]
        # Add the style indicator
        if style_char == 'A':
            style_indicator = A_SECTION_TOKEN
        elif style_char == 'B':
            style_indicator = B_SECTION_TOKEN
        elif style_char == 'C':
            style_indicator = C_SECTION_TOKEN
        elif style_char == 'D':
            style_indicator = D_SECTION_TOKEN
        else:
            style_indicator = None

        prompt_tokens.append(style_indicator)
        prompt.append(tokenizer.encode([style_indicator])[0])

        # Add tokens for this style, limited by snippet_length
        if style_idx in style_token_groups and len(style_token_groups[style_idx]) > 0:
            # Get the first group of tokens for this style
            tokens_for_style = style_token_groups[style_idx][0]
            tokens_to_add = tokens_for_style[:snippet_length]
            prompt.extend(tokens_to_add)
            prompt_tokens.extend([tokenizer.decode([t])[0] for t in tokens_to_add])
            # Pad if not enough tokens in group
            if len(tokens_to_add) < snippet_length:
                pad_amt = snippet_length - len(tokens_to_add)
                prompt.extend([tokenizer.pad_id] * pad_amt)
                prompt_tokens.extend([tokenizer.decode([tokenizer.pad_id])[0]] * pad_amt)
        else:
            # If no tokens for this style, pad with pad_id
            prompt.extend([tokenizer.pad_id] * snippet_length)
            prompt_tokens.extend([tokenizer.decode([tokenizer.pad_id])[0]] * snippet_length)

    # Add and encode the end indicator
    prompt_tokens.append(PROMPT_END_TOKEN)
    prompt.append(tokenizer.encode([PROMPT_END_TOKEN])[0])

    # Truncate if over max_prompt_length (shouldn't happen, but just in case)
    if len(prompt) > max_prompt_length:
        prompt = prompt[:max_prompt_length-1] + [tokenizer.encode([PROMPT_END_TOKEN])[0]]
    if len(prompt_tokens) > max_prompt_length:
        prompt_tokens = prompt_tokens[:max_prompt_length-1] + [PROMPT_END_TOKEN]

    return prompt, prompt_tokens

def get_batch_prompts_from_midi_style_ids(input_tokens_batch, style_ids_batch, tokenizer, max_prompt_length=512):
    """
    Construct prompts from batched MIDI tokens and style IDs using max_prompt_length.
    Each prompt is in the format:
    <PROMPT_START> <music_style_token> <A_SECTION> (snippet) <B_SECTION> (snippet) ... <PROMPT_END>
    
    Args:
        input_tokens_batch: Batch of input token IDs (batch_size x seq_length)
        style_ids_batch: Batch of style IDs (batch_size x seq_length)
        tokenizer: MusicTokenizerWithStyle instance
        max_prompt_length: Maximum total number of tokens in the prompt (including all indicators and snippets)
        
    Returns:
        Tuple of (batch_prompt_token_ids, batch_prompt_token_strings)
    """
    batch_size = input_tokens_batch.shape[0]
    batch_prompts = []
    batch_prompt_tokens = []
    
    for batch_idx in range(batch_size):
        # Process each example in the batch
        input_tokens = input_tokens_batch[batch_idx]
        style_ids = style_ids_batch[batch_idx]
        
        # Filter out padding tokens if present
        if hasattr(tokenizer, 'pad_id'):
            non_padding_mask = input_tokens != tokenizer.pad_id
            input_tokens = input_tokens[non_padding_mask]
            style_ids = style_ids[non_padding_mask]
        
        # Get prompt for this example, using max_prompt_length
        prompt, prompt_tokens = get_prompt_from_midi_style_ids(
            input_tokens, style_ids, tokenizer, max_prompt_length
        )
        
        batch_prompts.append(prompt)
        batch_prompt_tokens.append(prompt_tokens)
    
    # Convert to appropriate format (list of lists)
    return batch_prompts, batch_prompt_tokens

def get_random_midi(data_dir: str) -> str:
    """Get a random MIDI file from the given data directory."""
    midi_files = list(Path(data_dir).glob('**/*.mid'))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {data_dir}")
    return random.choice(midi_files)

def condense_style_sequence(labels):
    """Convert a sequence of style labels to a condensed format.
    
    Example: ['A', 'A', 'A', 'B', 'B', 'A'] -> 'Ax3, Bx2, Ax1'
    """
    if not labels:
        return ""
    
    condensed = []
    current_label = labels[0]
    count = 1
    
    for label in labels[1:]:
        if label == current_label:
            count += 1
        else:
            condensed.append(f"{current_label}x{count}")
            current_label = label
            count = 1
    
    # Add the last group
    condensed.append(f"{current_label}x{count}")
    
    return ", ".join(condensed)

def get_midi_ids_from_style_ids(midi_ids, style_ids, tokenizer):
    """Get MIDI IDs from style IDs."""
    midi_for_style = []
    for style_id in style_ids:
        midi_for_style.append(midi_ids[style_id])
    return midi_for_style

def extract_style_change_timestamps(tokens, style_tokens, tokenizer):
    """Get timestamps (in ms) for each major style change."""
    style_tokens = ''.join(style_tokens)
    if not style_tokens or len(style_tokens) <= 1:
        return []
    
    total_tokens = len(style_tokens)
    threshold = total_tokens * 0.05  # Groups below 5% will be filtered out

    # Group contiguous style tokens
    segments = []
    current_style = style_tokens[0]
    start_idx = 0
    for i in range(1, len(style_tokens)):
        if style_tokens[i] != current_style:
            segments.append((current_style, start_idx, i - 1))
            current_style = style_tokens[i]
            start_idx = i
    segments.append((current_style, start_idx, len(style_tokens) - 1))

    # Remove small segments
    significant_segments = []
    for seg in segments:
        style, start, end = seg
        if (end - start + 1) >= threshold:
            significant_segments.append(seg)
    
    if not significant_segments:
        return []
    
    # Determine transition indices
    change_indices = []
    for i in range(1, len(significant_segments)):
        change_indices.append(significant_segments[i][1])

    try:
        # Compute timestamps at these change indices
        timestamps = []
        for idx in change_indices:
            try:
                # Get timestamp from the tokens directly
                decoded_tokens = tokenizer.decode(tokens[:idx+1])
                timestamp_ms = tokenizer.calc_length_ms(decoded_tokens, onset=False)
                minutes = timestamp_ms // 60000
                seconds = (timestamp_ms % 60000) / 1000
                time_str = f"{minutes}:{seconds:06.3f}" if minutes else f"{seconds:.3f}s"
                timestamps.append((style_tokens[idx], time_str))
            except Exception as e:
                timestamps.append((style_tokens[idx], None))
        
        return timestamps
    except Exception as e:
        return [(style_tokens[idx], None) for idx in change_indices]

def filter_significant_styles(style_tokens):
    """Get style tokens with groups below 5% filtered out, then expand to match original length. Also convert to relative labels.
    Example: Ax1000, Bx1, Ax78, Bx1, Ax5, Bx1, Ax2, Bx1, Ax8, Bx2, 
    Ax1, Bx2, Ax1, Bx4, Ax2, Bx1, Ax2, Bx2, Ax1, Bx1, Ax4, Bx2, Ax2, 
    Bx1, Ax1, Bx2, Ax2, Bx1, Ax1, Bx2, Ax1, Bx2, Ax2, Bx1, Ax1, Bx2, 
    Ax1, Bx11, Ax1, Bx2, Ax1, Bx492, Ax1, Bx2, Ax1, Bx12, Ax1, Bx5, 
    Ax1, Bx2, Ax1, Bx4, Ax2, Bx2, Ax1, Bx1
    
    -> ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A']
    
    """
    if not style_tokens or len(style_tokens) <= 1:
        return []
    
    total_tokens = len(style_tokens)
    threshold = total_tokens * 0.05  # Groups below 5% will be filtered out

    # Group contiguous style tokens
    segments = []
    current_style = style_tokens[0]
    start_idx = 0
    for i in range(1, len(style_tokens)):
        if style_tokens[i] != current_style:
            segments.append((current_style, start_idx, i - 1))
            current_style = style_tokens[i]
            start_idx = i
    segments.append((current_style, start_idx, len(style_tokens) - 1))

    # Remove small segments
    significant_segments = []
    for seg in segments:
        style, start, end = seg
        if (end - start + 1) >= threshold:
            significant_segments.append(seg)
    
    if not significant_segments:
        return []
    
    # Reconstruct the style tokens with only significant segments
    significant_labels = []
    for seg in significant_segments:
        style, start, end = seg
        significant_labels.extend([style] * (end - start + 1))
    
    # If the length doesn't match the original, adjust to match original length
    if len(significant_labels) < total_tokens:
        # Calculate how many tokens to add
        tokens_to_add = total_tokens - len(significant_labels)
        
        # Distribute tokens proportionally across significant segments
        if significant_segments:
            tokens_per_segment = tokens_to_add // len(significant_segments)
            remainder = tokens_to_add % len(significant_segments)
            
            expanded_labels = []
            for i, seg in enumerate(significant_segments):
                style, _, _ = seg
                # Calculate extra tokens for this segment
                extra = tokens_per_segment + (1 if i < remainder else 0)
                # Add the original segment tokens
                segment_length = significant_labels.count(style)
                expanded_labels.extend([style] * segment_length)
                # Add extra tokens to match original length
                expanded_labels.extend([style] * extra)
            
            significant_labels = expanded_labels[:total_tokens]
    
    relative_labels = relative_label_sequence(significant_labels)
    return relative_labels

def relative_label_sequence(style_seq):
    """
    Convert a style sequence (e.g., ['B', 'C', 'D', 'B', 'C']) to a relative sequence
    where the first unique style is 'A', the next is 'B', etc.
    Example: ['B', 'C', 'D', 'B', 'C'] -> ['A', 'B', 'C', 'A', 'B']
    """
    if not style_seq:
        return []
    mapping = {}
    next_label_ord = ord('A')
    rel_seq = []
    for s in style_seq:
        if s not in mapping:
            mapping[s] = chr(next_label_ord)
            next_label_ord += 1
        rel_seq.append(mapping[s])
    return rel_seq

def get_music_style_from_condensed_style_sequence(significant_labels):
    """
    Extracts the music style from a list of style tokens and (optionally) tracks counts of each style.

    Args:
        significant_labels: List of style tokens (e.g., ['A', 'A', 'B', 'B', 'B', 'C'])

    Returns:
        A string containing just the style letters (e.g., "ABC")
    """
    if not significant_labels:
        return ""

    # First condense the sequence
    significant_condensed = condense_style_sequence(significant_labels)

    # Extract just the style letters (A, B, C, D) from the condensed representation
    import re
    styles = re.findall(r'([A-D])x\d+', significant_condensed)

    # Join the styles into a single string
    style_str = "".join(styles)

    # --- Parallel-safe tracking: append-only, no locks, no in-place updates ---
    # Instead of updating a shared file, we write out a single line per call to a log file.
    # This can be post-processed later to get unique styles and counts.

    try:
        # Write the style string to a log file (append-only, no locking)
        # Each process just appends its result; deduplication/counting is done offline.
        styles_log_path = "music_styles_log.txt"
        if style_str:
            with open(styles_log_path, "a") as f:
                f.write(f"{style_str}\n")

    except Exception as e:
        logger.warning(f"Failed to log music style: {e}")

    return style_str

    
def get_style_labels(label_path: str) -> Optional[List[str]]:
    """Reads a style file and returns the style characters."""
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()
        if not content:
            logger.warning(f"Empty style file: {label_path}")
            return None

        # Return the raw characters as a list
        return list(content)
    except Exception as e:
        logger.error(f"Error processing style file {label_path}: {e}", exc_info=False)
        return None

def process_synthetic_pair(args: Tuple[str, str, str], tokenizer: MusicTokenizerWithStyle) -> Optional[Tuple[List[str], List[str], str]]:
    """Processes a single (MIDI path, Style path) pair."""
    midi_path, label_path = args
    midi_tokens = tokenizer.tokenize_from_file(midi_path)
    style_labels = get_style_labels(label_path)

    if midi_tokens is None or style_labels is None:
        return None

    if len(midi_tokens) != len(style_labels):
        logger.warning(f"Length mismatch: MIDI={len(midi_tokens)}, Style={len(style_labels)} for {Path(midi_path).name}. Skipping.")
        return None

    if len(midi_tokens) == 0:
         logger.warning(f"Zero length sequence found for {Path(midi_path).name}. Skipping.")
         return None

    return midi_tokens, style_labels, midi_path

def process_real_midi(midi_path: str, tokenizer: MusicTokenizerWithStyle) -> Optional[Tuple[List[str], str]]:
    """Processes a single real MIDI path."""
    midi_tokens = tokenizer.tokenize_from_file(midi_path)
    if midi_tokens is None or len(midi_tokens) == 0:
        return None
    
    return midi_tokens, midi_path

    
def load_data_parallel(
    data_dir: str,
    tokenizer,
    max_len: int,
    mode: str = "synthetic",
    dataset_seq_limit: Optional[int] = None,
    cache_dir: str = "./cache",
    shuffle: bool = True,
    skip_long_sequences: bool = True
) -> List[Dict]:
    """Loads and processes data in parallel with caching."""
    num_workers = int(cpu_count() / 2)
    cache_path = None

    # --- cache naming based on params -------------------------
    os.makedirs(cache_dir, exist_ok=True)
    # Extract the directory name from data_dir for more readable cache files
    data_dir_name = os.path.basename(os.path.normpath(data_dir))
    params = f"{data_dir}_{tokenizer.vocab_size}_{max_len}_{mode}_{dataset_seq_limit}_{shuffle}_{skip_long_sequences}"
    h = hashlib.md5(params.encode()).hexdigest()[:8]
    # Include data_dir name and seq limit in the filename for better identification
    fname = f"{mode}_{data_dir_name}_max{max_len}_limit{dataset_seq_limit}_{h}.pkl"
    cache_path = os.path.join(cache_dir, fname)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            os.remove(cache_path)

    logger = logging.getLogger(__name__)
    logger.info(f"Processing {mode} data from scratch ({data_dir}) with {num_workers} workers")

    final_data: List[Dict] = []
    batch_size = 100000

    if mode == "synthetic":
        midi_dir  = os.path.join(data_dir, "midi")
        style_dir = os.path.join(data_dir, "style")
        if not os.path.isdir(midi_dir) or not os.path.isdir(style_dir):
            raise FileNotFoundError(f"Missing 'midi' or 'style' in {data_dir}")

        midi_files  = {p.stem: p for p in Path(midi_dir).glob("*.mid")}
        style_files = {p.stem: p for p in Path(style_dir).glob("*.txt")}
        stems = sorted(set(midi_files) & set(style_files))
        pairs = [(str(midi_files[s]), str(style_files[s])) for s in stems]
        # Cache file paths if available
        paths_cache_path = os.path.join(cache_dir, f"paths_{data_dir_name}_{h}.pkl")
        if os.path.exists(paths_cache_path):
            try:
                with open(paths_cache_path, "rb") as f:
                    pairs = pickle.load(f)
                logger.info(f"Loaded {len(pairs)} file pairs from cache")
            except Exception as e:
                logger.warning(f"Failed to load paths from cache: {e}")
                # If loading fails, continue with the original pairs
        else:
            # Save the paths for future use
            try:
                with open(paths_cache_path, "wb") as f:
                    pickle.dump(pairs, f)
                logger.info(f"Cached {len(pairs)} file pairs")
            except Exception as e:
                logger.warning(f"Failed to cache paths: {e}")
        if shuffle: random.shuffle(pairs)
        logger.info(f"Found {len(pairs)} synthetic pairs")

        for start in range(0, len(pairs), batch_size):
            if dataset_seq_limit and len(final_data) >= dataset_seq_limit:
                break
            batch = pairs[start:start+batch_size]
            logger.info(f"Processing batch {start//batch_size+1} ({len(batch)} files)")
            with Pool(num_workers) as pool:
                results = pool.imap_unordered(
                    partial(process_synthetic_pair, tokenizer=tokenizer),
                    batch
                )

                skipped = 0
                for midi_tokens, style_labels, file_path in tqdm(results, total=len(batch), desc="parsing"):
                    L = len(midi_tokens)
                    # truncate if needed
                    if L > max_len:
                        if skip_long_sequences:
                            skipped += 1
                            continue
                        midi_tokens = midi_tokens[:max_len]
                        style_labels = style_labels[:max_len]

                    final_data.append({
                        "tokens": midi_tokens,
                        "style_labels": style_labels,
                        "file_path": file_path
                    })
                    if dataset_seq_limit and len(final_data) >= dataset_seq_limit:
                        break

                if skipped:
                    logger.info(f"Skipped {skipped} overlong sequences")
                logger.info(f"Dataset size so far: {len(final_data)}")

    elif mode == "real":
        # Define paths cache file for real MIDI files
        paths_cache_path = os.path.join(cache_dir, f"{mode}_{data_dir_name}_paths_cache.pkl")
        
        # Try to load paths from cache
        if os.path.exists(paths_cache_path):
            try:
                with open(paths_cache_path, "rb") as f:
                    all_midis = pickle.load(f)
                logger.info(f"Loaded {len(all_midis)} real MIDI file paths from cache")
            except Exception as e:
                logger.warning(f"Failed to load real MIDI paths from cache: {e}")
                # If loading fails, find paths again
                all_midis = []
                for ext in ("*.mid","*.midi"):
                    all_midis += list(Path(data_dir).glob(f"**/{ext}"))
                logger.info(f"Found {len(all_midis)} real MIDI files")
        else:
            # Find all MIDI files
            all_midis = []
            for ext in ("*.mid","*.midi"):
                all_midis += list(Path(data_dir).glob(f"**/{ext}"))
            logger.info(f"Found {len(all_midis)} real MIDI files")
            
            # Save the paths for future use
            try:
                with open(paths_cache_path, "wb") as f:
                    pickle.dump(all_midis, f)
                logger.info(f"Cached {len(all_midis)} real MIDI file paths")
            except Exception as e:
                logger.warning(f"Failed to cache real MIDI paths: {e}")
                
        if shuffle: random.shuffle(all_midis)

        for start in range(0, len(all_midis), batch_size):
            if dataset_seq_limit and len(final_data) >= dataset_seq_limit:
                break
            batch = [str(p) for p in all_midis[start:start+batch_size]]
            logger.info(f"Processing real batch {start//batch_size+1} ({len(batch)})")
            with Pool(num_workers) as pool:
                results = pool.imap_unordered(
                    partial(process_real_midi, tokenizer=tokenizer),
                    batch
                )
                skipped = 0
                for midi_tokens, file_path in tqdm(results, total=len(batch), desc="parsing"):
                    L = len(midi_tokens)
                    if L > max_len:
                        if skip_long_sequences:
                            skipped += 1
                            continue
                        midi_tokens = midi_tokens[:max_len]

                    final_data.append({
                        "tokens": midi_tokens,
                        "file_path": file_path
                    })
                    if dataset_seq_limit and len(final_data) >= dataset_seq_limit:
                        break

                if skipped:
                    logger.info(f"Skipped {skipped} overlong sequences")
                logger.info(f"Dataset size so far: {len(final_data)}")

    else:
        raise ValueError("mode must be 'synthetic' or 'real'")

    logger.info(f"Final dataset size ({mode}): {len(final_data)}")
    with open(cache_path, "wb") as f:
        pickle.dump(final_data, f)
    logger.info(f"Cached data to {cache_path}")
    return final_data