#!/usr/bin/env python
"""
Generation Script for Decoder-Only Baseline

This script loads a trained decoder-only model and generates music
conditioned on snippet prompts.

Usage:
    python scripts/generate_decoder_only.py --checkpoint path/to/model.ckpt --output_dir generated/
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import sys
import os
sys.path.append('.')

from staria.models.tokenizer import MusicTokenizerWithStyle
from staria.baselines.decoder_only import DecoderOnlyBaseline


def create_sample_prompts(tokenizer: MusicTokenizerWithStyle, num_samples: int = 5) -> List[List[int]]:
    """
    Create sample prompts for generation.
    
    These are example prompts that could be extracted from real MIDI files
    or created manually for testing different musical styles.
    """
    prompts = []
    
    # Sample prompt structures
    prompt_templates = [
        # Simple AB structure
        ["<PROMPT_START>", "<A_SECTION>", "<B_SECTION>", "<PROMPT_END>"],
        
        # ABC structure
        ["<PROMPT_START>", "<A_SECTION>", "<B_SECTION>", "<C_SECTION>", "<PROMPT_END>"],
        
        # ABA structure
        ["<PROMPT_START>", "<A_SECTION>", "<B_SECTION>", "<A_SECTION>", "<PROMPT_END>"],
        
        # ABAB structure
        ["<PROMPT_START>", "<A_SECTION>", "<B_SECTION>", "<A_SECTION>", "<B_SECTION>", "<PROMPT_END>"],
        
        # ABCD structure
        ["<PROMPT_START>", "<A_SECTION>", "<B_SECTION>", "<C_SECTION>", "<D_SECTION>", "<PROMPT_END>"],
    ]
    
    # Create placeholder snippet tokens (in practice these would be real musical content)
    placeholder_snippet = ["<PLACEHOLDER_MUSIC>"] * 32  # Placeholder for actual music tokens
    
    for i in range(min(num_samples, len(prompt_templates))):
        template = prompt_templates[i]
        prompt_tokens = []
        
        for token in template:
            prompt_tokens.append(token)
            # Add placeholder snippet after each section token
            if token.endswith("_SECTION>"):
                prompt_tokens.extend(placeholder_snippet)
        
        # Encode to IDs
        try:
            prompt_ids = tokenizer.encode(prompt_tokens)
            prompts.append(prompt_ids)
        except Exception as e:
            logging.warning(f"Could not encode prompt template {i}: {e}")
            # Create minimal prompt as fallback
            minimal_prompt = tokenizer.encode(["<PROMPT_START>", "<PROMPT_END>"])
            prompts.append(minimal_prompt)
    
    return prompts


def generate_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 5,
    max_new_tokens: int = 1000,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
) -> None:
    """Generate music samples from a trained decoder-only model."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = MusicTokenizerWithStyle()
    
    # Load model from checkpoint
    logging.info(f"Loading model from {checkpoint_path}...")
    model = DecoderOnlyBaseline.load_from_checkpoint(
        checkpoint_path,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logging.info(f"Model loaded on device: {device}")
    
    # Create sample prompts
    logging.info("Creating sample prompts...")
    prompt_lists = create_sample_prompts(tokenizer, num_samples)
    
    # Generate samples
    logging.info(f"Generating {len(prompt_lists)} samples...")
    
    for i, prompt_ids in enumerate(prompt_lists):
        try:
            # Convert to tensor and move to device
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
            
            logging.info(f"Generating sample {i+1}/{len(prompt_lists)} (prompt length: {len(prompt_ids)})")
            
            # Generate sequence
            with torch.no_grad():
                generated = model.generate(
                    prompt_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            
            # Convert back to tokens
            generated_ids = generated[0].cpu().tolist()
            generated_tokens = tokenizer.decode(generated_ids)
            
            # Save as text
            text_path = os.path.join(output_dir, f"generated_sample_{i+1}.txt")
            with open(text_path, "w") as f:
                f.write(f"Generated sample {i+1}\n")
                f.write(f"Prompt length: {len(prompt_ids)} tokens\n")
                f.write(f"Generated length: {len(generated_ids)} tokens\n")
                f.write(f"Total length: {len(generated_ids)} tokens\n\n")
                f.write("Generated sequence:\n")
                f.write(str(generated_tokens))
            
            logging.info(f"Saved text to {text_path}")
            
            # Try to save as MIDI
            try:
                midi_path = os.path.join(output_dir, f"generated_sample_{i+1}.mid")
                tokenizer.ids_to_file(generated_ids, midi_path)
                logging.info(f"Saved MIDI to {midi_path}")
            except Exception as e:
                logging.warning(f"Could not save MIDI for sample {i+1}: {e}")
                
        except Exception as e:
            logging.error(f"Error generating sample {i+1}: {e}")
            continue
    
    logging.info(f"Generation complete. Samples saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate music with decoder-only baseline")
    
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="generated_decoder_baseline", 
                       help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Generate samples
    generate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )


if __name__ == "__main__":
    main()