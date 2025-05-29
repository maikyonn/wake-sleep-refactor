"""
MIDI Preprocessing Module

This module processes MIDI files and their style labels for music generation tasks.

Output PKL File Structure:
- The preprocessing creates .pkl files with the following structure:
  - 'tokens': List of tokenized MIDI sequences
  - 'style_labels': List of style labels corresponding to each token (only in synthetic mode)
  - 'file_paths': List of original file paths for reference
  - 'metadata': Dictionary containing preprocessing parameters:
    - 'max_len': Maximum sequence length
    - 'mode': 'synthetic' or 'real'
    - 'vocab_size': Size of the tokenizer vocabulary
    - 'dataset_name': Name of the processed dataset
    - 'timestamp': When the preprocessing was performed

Usage:
    python preprocess.py --data_dir /path/to/data --mode synthetic
"""
import argparse
import datetime
import numpy as np
import logging
import os
import sys
from pathlib import Path
from multiprocessing import cpu_count
from functools import partial
from multiprocessing import Pool
import random
import hashlib
import pickle
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.StariaTokenizer import MusicTokenizerWithStyle


def load_from_pkl(pkl_file: str) -> Dict:
    """
    Loads and unpacks a preprocessed PKL file.
    
    Args:
        pkl_file: Path to the PKL file to load
        
    Returns:
        Dictionary containing the unpacked data with keys:
        - 'tokens': List of tokenized MIDI sequences
        - 'file_paths': List of original file paths
        - 'style_labels': List of style labels (only in synthetic mode)
        - 'metadata': Dictionary of preprocessing parameters
    """
    with open(pkl_file, "rb") as f:
        import psutil, pickle, sys
        data = pickle.load(f)
        print(f'In‑RAM : {psutil.Process().memory_info().rss/2**30:.1f} GB')
    
    return data

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

def process_midi_subdir(subdir_path):
    """Process all MIDI files in a subdirectory."""
    midi_files = []
    for root, _, files in os.walk(subdir_path):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, f))
    return midi_files

def load_data_parallel(
    data_dir: str,
    tokenizer,
    max_len: int,
    mode: str = "synthetic",
    dataset_seq_limit: Optional[int] = None,
    cache_dir: str = "./cache",
    shuffle: bool = True,
    skip_long_sequences: bool = True
) -> Tuple[List[Dict], str]:
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
                return pickle.load(f), cache_path
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
        # Cache file paths with the same naming convention as the data cache
        paths_cache_path = os.path.join(cache_dir, f"paths_{mode}_{data_dir_name}_max{max_len}_limit{dataset_seq_limit}_{h}.pkl")
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
        # Define paths cache file for real MIDI files using the same naming convention as the data cache
        paths_cache_path = os.path.join(cache_dir, f"paths_{mode}_{data_dir_name}_max{max_len}_limit{dataset_seq_limit}_{h}.pkl")
        
        # Try to load paths from cache
        if os.path.exists(paths_cache_path):
            try:
                with open(paths_cache_path, "rb") as f:
                    all_midis = pickle.load(f)
                logger.info(f"Loaded {len(all_midis)} real MIDI file paths from cache")
            except Exception as e:
                logger.warning(f"Failed to load real MIDI paths from cache: {e}")
                # If loading fails, find paths again
                all_midis = find_midi_files(data_dir, num_workers, logger)
                
                # Save the paths for future use
                try:
                    with open(paths_cache_path, "wb") as f:
                        pickle.dump(all_midis, f)
                    logger.info(f"Cached {len(all_midis)} real MIDI file paths")
                except Exception as e:
                    logger.warning(f"Failed to cache real MIDI paths: {e}")
        else:
            # Find all MIDI files
            all_midis = find_midi_files(data_dir, num_workers, logger)
            
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
            batch = all_midis[start:start+batch_size]
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
    return final_data, cache_path

def find_midi_files(data_dir, num_workers, logger):
    """Find all MIDI files in the given directory, using parallel processing for subdirectories."""
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if subdirs:
        logger.info(f"Found {len(subdirs)} subdirectories, processing each in parallel")
        
        # Process each subdirectory in parallel to find MIDI files
        all_midis = []
        subdir_paths = [os.path.join(data_dir, subdir) for subdir in subdirs]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_midi_subdir, subdir_path) for subdir_path in subdir_paths]
            for future in as_completed(futures):
                try:
                    midi_files = future.result()
                    logger.info(f"Found {len(midi_files)} MIDI files in subdirectory")
                    all_midis.extend(midi_files)
                except Exception as e:
                    logger.error(f"Error processing subdirectory: {e}")
    else:
        # No subdirectories, just find all MIDI files directly
        all_midis = []
        for ext in ("*.mid", "*.midi"):
            all_midis.extend([str(p) for p in Path(data_dir).glob(f"**/{ext}")])
    
    logger.info(f"Found {len(all_midis)} real MIDI files")
    return all_midis

if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preprocess MIDI files for music generation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing MIDI files")
    parser.add_argument("--mode", type=str, choices=["synthetic", "real"], default="synthetic", 
                        help="Processing mode: 'synthetic' for paired MIDI+style, 'real' for MIDI only")
    parser.add_argument("--max_len", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--dataset_seq_limit", type=int, default=None, help="Limit number of sequences to process")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for caching intermediate results")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling of input files")
    parser.add_argument("--keep_long_sequences", default=False, help="Keep and truncate sequences longer than max_len")
    
    args = parser.parse_args()
    
    
    # Initialize tokenizer
    tokenizer = MusicTokenizerWithStyle()
    
    # Process the data
    logger.info(f"Starting preprocessing in {args.mode} mode")
    data, output_file = load_data_parallel(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_len=args.max_len,
        mode=args.mode,
        dataset_seq_limit=args.dataset_seq_limit,
        cache_dir=args.cache_dir,
        shuffle=not args.no_shuffle,
        skip_long_sequences=not args.keep_long_sequences
    )
    
    # Extract dataset name from data_dir
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))

    
    logger.info(f"Preprocessing complete. Saved {len(data)} sequences to {output_file}")
    # Load and print the first item from the saved PKL file
    try:
        saved_data = load_from_pkl(output_file)
        if saved_data and isinstance(saved_data, dict):
            # Print metadata
            logger.info(f"Metadata from saved file: {saved_data.get('metadata', {})}")
            
            # Print first token sequence
            tokens = saved_data.get('tokens', [])
            if tokens and len(tokens) > 0:
                logger.info(f"First token sequence (first 20 tokens): {tokens[0][:20]}")
            
            # Print first style label sequence if available
            style_labels = saved_data.get('style_labels', [])
            if style_labels and len(style_labels) > 0:
                logger.info(f"First style label sequence (first 20 labels): {style_labels[0][:20]}")
            
            # Print first file path
            file_paths = saved_data.get('file_paths', [])
            if file_paths and len(file_paths) > 0:
                logger.info(f"First file path: {file_paths[0]}")
        else:
            logger.warning("Could not load data from the saved PKL file or data format is unexpected")
    except Exception as e:
        logger.error(f"Error loading saved PKL file: {e}")