#!/usr/bin/env python
"""
Generate music using a trained Staria model.

This script loads a trained Staria model checkpoint and generates music sequences
from prompts or validation data.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Optional
import copy

import lightning as L
import torch
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
from src.ariautils.midi import MidiDict

from fabric_decoder import LitStaria, collate_fn, OnDemandMidiDataset

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
# Utility to build dataloaders
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloader(
    index_file: str,
    tokenizer: MusicTokenizerWithStyle,
    batch_size: int = 2,
    num_workers: int = 4,
    max_len: int = 4096,
    use_snippet: bool = True,
    seq_limit: Optional[int] = None,
):
    dataset = OnDemandMidiDataset(index_file, tokenizer, max_len=max_len, use_snippet=use_snippet, seq_limit=seq_limit)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    return dataloader


# ──────────────────────────────────────────────────────────────────────────────
# Generation utilities
# ──────────────────────────────────────────────────────────────────────────────

def combine_sequences_with_timing(tokenizer: MusicTokenizerWithStyle, 
                                 sequences: List[List[int]], 
                                 add_piano_prefix: bool = True) -> List[int]:
    """
    Combine multiple token sequences into a single MIDI file with proper timing.
    Based on the approach in create_synthetic_midi.py.
    
    Args:
        tokenizer: The tokenizer instance
        sequences: List of token sequences to combine
        add_piano_prefix: Whether to add piano prefix tokens (6, 0) at the start
    
    Returns:
        Combined token sequence that can be converted to MIDI
    """
    if not sequences:
        return []
    
    combined_midi = None
    current_end_time = 0
    
    for seq_idx, token_ids in enumerate(sequences):
        if not token_ids:
            continue
            
        try:
            # Remove piano prefix tokens if they exist at the beginning
            clean_tokens = token_ids.copy()
            if len(clean_tokens) >= 2 and clean_tokens[0] == 6 and clean_tokens[1] == 0:
                clean_tokens = clean_tokens[2:]
            
            # Convert tokens to string tokens, then to MIDI
            if clean_tokens and isinstance(clean_tokens[0], torch.Tensor):
                clean_tokens = [t.item() for t in clean_tokens]
            
            token_strings = tokenizer.decode(clean_tokens)
            if not token_strings:
                continue
                
            midi_dict = tokenizer._tokenizer.detokenize(token_strings)
            if not midi_dict or not midi_dict.note_msgs:
                continue
            
            # Normalize timing - subtract the earliest start time
            if midi_dict.note_msgs:
                min_start = min(msg["data"]["start"] for msg in midi_dict.note_msgs)
                for msg in midi_dict.note_msgs:
                    msg["tick"] -= min_start
                    msg["data"]["start"] -= min_start
                    msg["data"]["end"] -= min_start
            
            if seq_idx == 0:
                # First sequence - use as base
                combined_midi = midi_dict
            else:
                # Subsequent sequences - offset timing and append
                for msg in midi_dict.note_msgs:
                    new_msg = copy.deepcopy(msg)
                    new_msg["tick"] += current_end_time
                    new_msg["data"]["start"] += current_end_time
                    new_msg["data"]["end"] += current_end_time
                    combined_midi.note_msgs.append(new_msg)
            
            # Update current end time
            if combined_midi.note_msgs:
                current_end_time = max(msg["data"]["end"] for msg in combined_midi.note_msgs)
                
        except Exception as e:
            logger.warning(f"Error processing sequence {seq_idx}: {e}")
            continue
    
    if combined_midi is None or not combined_midi.note_msgs:
        return []
    
    # Convert back to tokens
    try:
        combined_tokens = tokenizer.tokenize(combined_midi)
        combined_ids = tokenizer.encode(combined_tokens)
        
        # Add piano prefix if requested
        if add_piano_prefix:
            combined_ids = [6, 0] + combined_ids
            
        return combined_ids
    except Exception as e:
        logger.error(f"Error converting combined MIDI back to tokens: {e}")
        return []


def generate_from_batch(model: LitStaria, batch: Dict[str, torch.Tensor], 
                       max_new_tokens: int = 256, temperature: float = 0.7,
                       output_dir: str = "generated_samples"):
    """
    Generate music sequences from a batch of data.
    
    Args:
        model: The trained LitStaria model
        batch: Batch from dataloader containing encoder_ids/input_ids
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature for generation
        output_dir: Directory to save generated sequences
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        if "encoder_ids" in batch:
            # Snippet mode: use encoder_ids as context and first few tokens of decoder_ids as prompt
            context_ids = batch["encoder_ids"]
            decoder_ids = batch["decoder_ids"]
            
            # Use first 32 tokens as prompt
            prompt_length = min(32, decoder_ids.size(1))
            prompt_ids = decoder_ids[:, :prompt_length]
            
            logger.info(f"Generating with snippet mode - context shape: {context_ids.shape}, prompt shape: {prompt_ids.shape}")
            
            generated = model.generate(
                prompt_ids=prompt_ids,
                context_ids=context_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        else:
            # Autoregressive mode: use first few tokens as prompt
            input_ids = batch["input_ids"]
            prompt_length = min(32, input_ids.size(1))
            prompt_ids = input_ids[:, :prompt_length]
            
            logger.info(f"Generating with autoregressive mode - prompt shape: {prompt_ids.shape}")
            
            generated = model.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
    # Prepend tokens 6 and 0 to each generated sequence
    batch_size = generated.size(0)
    prefix_tokens = torch.tensor([6, 0], device=generated.device, dtype=generated.dtype)
    prefix_batch = prefix_tokens.unsqueeze(0).expand(batch_size, -1)
    generated = torch.cat([prefix_batch, generated], dim=1)
    
    # Also prepare original input sequences with prefix for comparison
    if "encoder_ids" in batch:
        original_ids = batch["decoder_ids"]
    else:
        original_ids = batch["input_ids"]
    
    # Add prefix to original sequences too
    original_with_prefix = torch.cat([prefix_batch, original_ids], dim=1)
    
    # Save generated sequences
    batch_size = generated.size(0)
    for i in range(batch_size):
        sequence = generated[i].cpu().numpy()
        original_sequence = original_with_prefix[i].cpu().numpy()
        
        # Decode tokens back to music tokens
        try:
            decoded_tokens = model.tokenizer.decode(sequence.tolist())
            print(f"Generated sample {i}:")
            print(decoded_tokens)
            
            # Decode original sequence for comparison
            original_decoded_tokens = model.tokenizer.decode(original_sequence.tolist())
            print(f"Original sample {i}:")
            print(original_decoded_tokens)
            
            # Save as text file
            output_path = os.path.join(output_dir, f"generated_sample_{i}.txt")
            with open(output_path, "w") as f:
                f.write(f"Generated sequence (length: {len(decoded_tokens)}):\n")
                f.write(str(decoded_tokens))
                f.write(f"\n\nOriginal sequence (length: {len(original_decoded_tokens)}):\n")
                f.write(str(original_decoded_tokens))
            
            logger.info(f"Saved generated sample {i} to {output_path}")
            
            # Try to convert to MIDI if possible
            try:
                midi_path = os.path.join(output_dir, f"generated_sample_{i}.mid")
                model.tokenizer.ids_to_file(sequence.tolist(), midi_path)
                logger.info(f"Saved MIDI file to {midi_path}")
                
                # Also save original as MIDI for comparison
                original_midi_path = os.path.join(output_dir, f"original_sample_{i}.mid")
                model.tokenizer.ids_to_file(original_sequence.tolist(), original_midi_path)
                logger.info(f"Saved original MIDI file to {original_midi_path}")
                
                # Create combined MIDI with proper timing if we have encoder context
                if "encoder_ids" in batch:
                    try:
                        encoder_sequence = batch["encoder_ids"][i].cpu().numpy()
                        decoder_sequence = batch["decoder_ids"][i].cpu().numpy()
                        
                        # Prepare sequences to combine: encoder input, decoder prompt, generated output
                        sequences_to_combine = []
                        
                        # Add encoder sequence (snippets)
                        if len(encoder_sequence) > 0:
                            sequences_to_combine.append(encoder_sequence.tolist())
                        
                        # Add decoder prompt (first part of decoder sequence)
                        prompt_length = min(32, len(decoder_sequence))
                        if prompt_length > 0:
                            sequences_to_combine.append(decoder_sequence[:prompt_length].tolist())
                        
                        # Add generated output (without the prefix that was added)
                        generated_clean = sequence[2:].tolist() if len(sequence) > 2 else sequence.tolist()
                        if len(generated_clean) > 0:
                            sequences_to_combine.append(generated_clean)
                        
                        # Combine with proper timing
                        combined_ids = combine_sequences_with_timing(
                            model.tokenizer, 
                            sequences_to_combine, 
                            add_piano_prefix=True
                        )
                        
                        if combined_ids:
                            combined_midi_path = os.path.join(output_dir, f"combined_sample_{i}.mid")
                            model.tokenizer.ids_to_file(combined_ids, combined_midi_path)
                            logger.info(f"Saved combined MIDI file to {combined_midi_path}")
                            
                            # Also save text representation of combined sequence
                            combined_text_path = os.path.join(output_dir, f"combined_sample_{i}.txt")
                            combined_tokens = model.tokenizer.decode(combined_ids)
                            with open(combined_text_path, "w") as f:
                                f.write(f"Combined sequence (encoder + prompt + generated, length: {len(combined_tokens)}):\n")
                                f.write(str(combined_tokens))
                            logger.info(f"Saved combined text to {combined_text_path}")
                    except Exception as e:
                        logger.warning(f"Could not create combined MIDI for sample {i}: {e}")
                        
            except Exception as e:
                logger.warning(f"Could not convert to MIDI for sample {i}: {e}")
                
        except Exception as e:
            logger.error(f"Could not decode generated sequence for sample {i}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI & Generation entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate music with trained Staria model")
    p.add_argument("--checkpoint", default='checkpoints/staria/last.ckpt', help="Path to model checkpoint")
    p.add_argument("--output_dir", default="generated_samples", help="Directory to save generated samples")
    p.add_argument("--max_new_tokens", type=int, default=4000, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.9, help="Temperature for generation")
    p.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate from scratch")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size for generation from data")
    
    # Data-based generation
    p.add_argument("--data_index", default='cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_val.pkl', help="Path to .pkl index file for data-based generation")
    p.add_argument("--use_snippet", action="store_true", default=True, help="Use snippet mode")
    p.add_argument("--data_limit", default=1,type=int, help="Limit number of data samples to use")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Initialize tokenizer
    tokenizer = MusicTokenizerWithStyle()
    
    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = LitStaria.load_from_checkpoint(
        args.checkpoint,
        tokenizer=tokenizer,
        use_snippet=args.use_snippet
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on device: {device}")
    logger.info(f"Model uses snippet mode: {model.use_snippet}")
    
    # Generate from data if provided
    if args.data_index:
        logger.info("Generating samples from data...")
        
        dataloader = make_dataloader(
            index_file=args.data_index,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            use_snippet=args.use_snippet,
            seq_limit=args.data_limit
        )
        
        # Get a sample batch
        data_iter = iter(dataloader)
        sample_batch = next(data_iter)
        
        if sample_batch is not None:
            # Move batch to device
            if "encoder_ids" in sample_batch:
                sample_batch["encoder_ids"] = sample_batch["encoder_ids"].to(device)
                sample_batch["decoder_ids"] = sample_batch["decoder_ids"].to(device)
            else:
                sample_batch["input_ids"] = sample_batch["input_ids"].to(device)
            
            # Generate samples
            generate_from_batch(
                model=model,
                batch=sample_batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                output_dir=args.output_dir
            )
        else:
            logger.warning("Could not get a valid batch from dataloader")
    
    logger.info(f"Generation complete. Samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
