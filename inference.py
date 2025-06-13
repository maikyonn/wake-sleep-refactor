#!/usr/bin/env python
"""
Run inference on Staria music-generation model.

This script loads a trained checkpoint and runs inference on samples from a dataset,
generating music tokens and optionally saving the results.

Usage:
    python inference.py --model_path checkpoints/best-epoch=050-val_loss=2.3456.ckpt \
                       --dataset_path /path/to/dataset \
                       --num_samples 10 \
                       --output_dir inference_results
"""

from __future__ import annotations
import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import pickle
from datetime import datetime
from tqdm import tqdm

import torch
import lightning as L
from torch.utils.data import DataLoader

# Import from the training script modules
from src.StariaTokenizer import (
    MusicTokenizerWithStyle,
    PROMPT_START_TOKEN, PROMPT_END_TOKEN,
    A_SECTION_TOKEN, B_SECTION_TOKEN, C_SECTION_TOKEN, D_SECTION_TOKEN,
    FORM_LABEL_MAP, IGNORE_LABEL_IDX, SPECIAL_TOKENS
)
from src.ariautils.midi import MidiDict
from src.utils_new import music_style_from_labels
from fabric_xt_2 import (
    LitXTransformer,
    OnDemandMidiDataset,
    midi_collate_mapstyle,
    custom_generate_top_p,
    inference_on_sample,
    _SNIPPET_LEN
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[LitXTransformer, MusicTokenizerWithStyle]:
    """Load a trained model from checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize tokenizer
    tokenizer = MusicTokenizerWithStyle()
    
    # Load the model from checkpoint
    model = LitXTransformer.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        map_location=device
    )
    
    model.eval()
    model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded successfully with {total_params:,} parameters")
    
    return model, tokenizer


def extract_snippets_from_encoder(encoder_tokens):
    """Extract snippet sections from encoder tokens."""
    snippets = {}
    current_section = None
    current_snippet = []
    
    for i, tok in enumerate(encoder_tokens):
        if tok == PROMPT_START_TOKEN:
            continue
        elif tok == PROMPT_END_TOKEN:
            if current_section:
                snippets[current_section] = current_snippet
            break
        elif tok in [A_SECTION_TOKEN, B_SECTION_TOKEN, C_SECTION_TOKEN, D_SECTION_TOKEN]:
            if current_section:
                snippets[current_section] = current_snippet
            current_section = tok[1]  # Extract 'A', 'B', 'C', or 'D'
            current_snippet = []
        else:
            if current_section:
                current_snippet.append(tok)
    
    return snippets


def merge_snippets_with_output(snippets, output_tokens, tokenizer):
    """
    Merge snippets with output tokens to create a complete MIDI.
    """
    merged_tokens = []
    
    # Extract prefix tokens from output
    prefix_tokens = []
    output_start_idx = 0
    for i, tok in enumerate(output_tokens):
        if isinstance(tok, tuple) and tok[0] == 'prefix':
            prefix_tokens.append(tok)
        elif tok == '<S>':
            prefix_tokens.append(tok)
            output_start_idx = i + 1
            break
    
    # Start with prefix tokens
    merged_tokens.extend(prefix_tokens)
    
    # Add all snippets sequentially
    current_time_offset = 0
    for section in ['A', 'B', 'C', 'D']:
        if section in snippets:
            snippet = snippets[section]
            # Track time tokens in snippet
            for tok in snippet:
                if tok == '<T>':
                    current_time_offset += 5000  # abs_time_step_ms
                merged_tokens.append(tok)
    
    # Add time separator between snippets and output
    if current_time_offset > 0:
        # Add one more time token to separate
        merged_tokens.append('<T>')
        current_time_offset += 5000
    
    # Process output tokens, adjusting onset times
    output_time_offset = 0
    for tok in output_tokens[output_start_idx:]:
        if tok == '<E>':
            continue  # Skip end token for now
        elif tok == '<T>':
            output_time_offset += 5000
            merged_tokens.append(tok)
        elif isinstance(tok, tuple) and tok[0] == 'onset':
            # Keep onset as is - it's relative to the current time segment
            merged_tokens.append(tok)
        else:
            merged_tokens.append(tok)
    
    # Add end token
    merged_tokens.append('<E>')
    
    return merged_tokens


def save_midi_files_with_snippets(sample, tokenizer, generated_tokens, target_tokens, 
                                  encoder_tokens, output_dir, idx, timestamp):
    """Save two MIDI files: snippets+target and snippets+generated."""
    try:
        # Extract snippets from encoder
        snippets = extract_snippets_from_encoder(encoder_tokens)
        
        if not snippets:
            logger.warning(f"No snippets found for sample {idx}")
            return False
        
        # Create merged tokens for target
        merged_target_tokens = merge_snippets_with_output(snippets, target_tokens, tokenizer)
        
        # Create merged tokens for generated
        merged_generated_tokens = merge_snippets_with_output(snippets, generated_tokens, tokenizer)
        
        # Save target+snippets MIDI
        try:
            target_midi_dict = tokenizer._tokenizer.detokenize(merged_target_tokens)
            target_midi = target_midi_dict.to_midi()
            target_midi_file = os.path.join(output_dir, f"sample_{idx:04d}_snippets_target_{timestamp}.mid")
            target_midi.save(target_midi_file)
            logger.info(f"Saved snippets+target MIDI: {target_midi_file}")
        except Exception as e:
            logger.warning(f"Failed to save snippets+target MIDI for sample {idx}: {e}")
        
        # Save generated+snippets MIDI
        try:
            generated_midi_dict = tokenizer._tokenizer.detokenize(merged_generated_tokens)
            generated_midi = generated_midi_dict.to_midi()
            generated_midi_file = os.path.join(output_dir, f"sample_{idx:04d}_snippets_generated_{timestamp}.mid")
            generated_midi.save(generated_midi_file)
            logger.info(f"Saved snippets+generated MIDI: {generated_midi_file}")
        except Exception as e:
            logger.warning(f"Failed to save snippets+generated MIDI for sample {idx}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in save_midi_files_with_snippets for sample {idx}: {e}")
        return False


def run_inference_on_dataset(
    model: LitXTransformer,
    tokenizer: MusicTokenizerWithStyle,
    dataset_path: Optional[str] = None,
    pkl_file: Optional[str] = None,
    num_samples: int = 10,
    max_length: int = 1024,
    output_dir: str = "inference_results",
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    save_midi: bool = False,
    use_tqdm_generation: bool = True
) -> List[Dict[str, Any]]:
    """Run inference on multiple samples from a dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = OnDemandMidiDataset(
        dataset_path=dataset_path,
        pkl_file=pkl_file,
        tokenizer=tokenizer,
        max_len=4096,
        use_snippet=True,
        max_dataset_length=num_samples  # Limit dataset size
    )
    
    if len(dataset) == 0:
        logger.error("Dataset is empty!")
        return []
    
    actual_samples = min(num_samples, len(dataset))
    logger.info(f"Running inference on {actual_samples} samples from dataset with {len(dataset)} total samples")
    
    results = []
    successful_inferences = 0
    
    # Progress bar for samples
    progress_bar = tqdm(range(actual_samples), desc="Processing samples")
    
    for idx in progress_bar:
        try:
            # Get sample from dataset
            sample = dataset[idx]
            if sample is None:
                logger.warning(f"Sample {idx} is None, skipping...")
                continue
            
            # Update progress description
            progress_bar.set_description(f"Processing sample {idx+1}/{actual_samples}")
            
            # Run inference
            logger.info(f"\nProcessing sample {idx+1}/{actual_samples}")
            
            # Extract input information
            encoder_ids = sample['encoder_ids'].tolist()
            decoder_ids = sample['decoder_ids'].tolist()
            
            encoder_tokens = tokenizer.decode(encoder_ids)
            target_tokens = tokenizer.decode(decoder_ids)
            
            # Generate output
            generated_tokens = inference_on_sample(
                model, 
                sample, 
                tokenizer, 
                max_length=max_length, 
                use_tqdm=use_tqdm_generation
            )
            
            # Calculate metrics
            if generated_tokens and target_tokens:
                match_count = sum(1 for g, t in zip(generated_tokens, target_tokens) if g == t)
                match_ratio = match_count / min(len(generated_tokens), len(target_tokens))
            else:
                match_count = 0
                match_ratio = 0.0
            
            # Store result
            result = {
                'sample_idx': idx,
                'encoder_length': len(encoder_tokens),
                'target_length': len(target_tokens),
                'generated_length': len(generated_tokens) if generated_tokens else 0,
                'encoder_tokens': encoder_tokens,
                'target_tokens': target_tokens,
                'generated_tokens': generated_tokens,
                'match_count': match_count,
                'match_ratio': match_ratio,
                'encoder_ids': encoder_ids,
                'target_ids': decoder_ids,
                'generated_ids': tokenizer.encode(generated_tokens) if generated_tokens else []
            }
            
            results.append(result)
            successful_inferences += 1
            
            # Save individual result
            sample_output_file = os.path.join(output_dir, f"sample_{idx:04d}_{timestamp}.txt")
            with open(sample_output_file, 'w') as f:
                f.write(f"SAMPLE {idx+1}/{actual_samples}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("ENCODER TOKENS (INPUT):\n")
                f.write("-" * 25 + "\n")
                f.write(f"Length: {result['encoder_length']}\n")
                f.write(f"Tokens: {result['encoder_tokens']}\n\n")
                
                f.write("TARGET TOKENS (EXPECTED OUTPUT):\n")
                f.write("-" * 35 + "\n")
                f.write(f"Length: {result['target_length']}\n")
                f.write(f"Tokens: {result['target_tokens']}\n\n")
                
                f.write("GENERATED TOKENS (ACTUAL OUTPUT):\n")
                f.write("-" * 35 + "\n")
                f.write(f"Length: {result['generated_length']}\n")
                f.write(f"Tokens: {result['generated_tokens']}\n\n")
                
                f.write("COMPARISON STATISTICS:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Matching tokens: {result['match_count']}\n")
                f.write(f"Match ratio: {result['match_ratio']:.3f} ({result['match_ratio']*100:.1f}%)\n")
                f.write(f"Generated/Target length ratio: {result['generated_length']/result['target_length']:.3f}\n")
            
            # Save MIDI files with snippets
            if save_midi and generated_tokens:
                save_midi_files_with_snippets(
                    sample=sample,
                    tokenizer=tokenizer,
                    generated_tokens=generated_tokens,
                    target_tokens=target_tokens,
                    encoder_tokens=encoder_tokens,
                    output_dir=output_dir,
                    idx=idx,
                    timestamp=timestamp
                )
            
            # Update progress bar
            progress_bar.set_postfix({
                'successful': successful_inferences,
                'match_ratio': f"{match_ratio:.2f}"
            })
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue
    
    progress_bar.close()
    
    # Save summary results
    summary_file = os.path.join(output_dir, f"inference_summary_{timestamp}.json")
    summary = {
        'timestamp': timestamp,
        'total_samples': actual_samples,
        'successful_inferences': successful_inferences,
        'average_match_ratio': sum(r['match_ratio'] for r in results) / len(results) if results else 0,
        'average_generated_length': sum(r['generated_length'] for r in results) / len(results) if results else 0,
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nInference complete!")
    logger.info(f"Processed {successful_inferences}/{actual_samples} samples successfully")
    logger.info(f"Average match ratio: {summary['average_match_ratio']:.3f}")
    logger.info(f"Results saved to: {output_dir}")
    
    return results


def run_inference_on_custom_prompt(
    model: LitXTransformer,
    tokenizer: MusicTokenizerWithStyle,
    prompt_tokens: List[str],
    max_length: int = 1024,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> List[str]:
    """Run inference on a custom prompt."""
    model.eval()
    model.to(device)
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt_tokens)
    encoder_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    encoder_mask = torch.ones_like(encoder_ids, dtype=torch.bool)
    
    # Create starting sequence for decoder
    seq_out_start = torch.tensor([[6]], device=device, dtype=torch.long)
    
    # Access the underlying XTransformer model
    underlying_model = model.model if hasattr(model, 'model') else model
    
    # Generate
    with torch.no_grad():
        generated = custom_generate_top_p(
            model=underlying_model,
            encoder_ids=encoder_ids,
            decoder_start=seq_out_start,
            max_length=max_length,
            encoder_mask=encoder_mask,
            use_tqdm=True,
            tokenizer=tokenizer
        )
    
    # Decode
    generated = generated.squeeze(0).cpu()
    generated_tokens = tokenizer.decode(generated.tolist())
    
    return generated_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on trained music transformer model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the model checkpoint (.ckpt file)")
    
    # Dataset arguments (one of these is required)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset_path", type=str, 
                             help="Path to dataset directory (with midi/ and style/ subdirectories)")
    dataset_group.add_argument("--pkl_file", type=str, 
                             help="Path to preprocessed pkl file containing dataset")
    
    # Optional arguments
    parser.add_argument("--num_samples", type=int, default=10, 
                       help="Number of samples to run inference on (default: 10)")
    parser.add_argument("--max_length", type=int, default=1024, 
                       help="Maximum length of generated sequence (default: 1024)")
    parser.add_argument("--output_dir", type=str, default="inference_results", 
                       help="Directory to save inference results (default: inference_results)")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["cpu", "cuda", "auto"],
                       help="Device to run inference on (default: auto)")
    parser.add_argument("--save_midi", action="store_true", 
                       help="Save generated sequences as MIDI files")
    parser.add_argument("--no_progress", action="store_true", 
                       help="Disable progress bars during generation")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility (default: 42)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    L.seed_everything(args.seed, workers=True)
    
    # Determine device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Check if model checkpoint exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint not found: {args.model_path}")
        return
    
    # Load model
    try:
        model, tokenizer = load_model_from_checkpoint(args.model_path, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Run inference
    try:
        results = run_inference_on_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            pkl_file=args.pkl_file,
            num_samples=args.num_samples,
            max_length=args.max_length,
            output_dir=args.output_dir,
            device=device,
            save_midi=args.save_midi,
            use_tqdm_generation=not args.no_progress
        )
        
        # Print summary statistics
        if results:
            logger.info("\n" + "="*50)
            logger.info("INFERENCE SUMMARY")
            logger.info("="*50)
            logger.info(f"Total samples processed: {len(results)}")
            
            avg_match_ratio = sum(r['match_ratio'] for r in results) / len(results)
            avg_generated_length = sum(r['generated_length'] for r in results) / len(results)
            avg_target_length = sum(r['target_length'] for r in results) / len(results)
            
            logger.info(f"Average match ratio: {avg_match_ratio:.3f} ({avg_match_ratio*100:.1f}%)")
            logger.info(f"Average generated length: {avg_generated_length:.1f} tokens")
            logger.info(f"Average target length: {avg_target_length:.1f} tokens")
            logger.info(f"Average length ratio: {avg_generated_length/avg_target_length:.3f}")
            
            # Find best and worst performing samples
            best_sample = max(results, key=lambda x: x['match_ratio'])
            worst_sample = min(results, key=lambda x: x['match_ratio'])
            
            logger.info(f"\nBest performing sample: #{best_sample['sample_idx']} (match ratio: {best_sample['match_ratio']:.3f})")
            logger.info(f"Worst performing sample: #{worst_sample['sample_idx']} (match ratio: {worst_sample['match_ratio']:.3f})")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()