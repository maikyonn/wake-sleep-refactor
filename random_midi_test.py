import os
import random
import shutil
import argparse
from pathlib import Path
import logging

def setup_logging():
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def find_midi_files(source_dir):
    """Recursively find all MIDI files in the source directory."""
    midi_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))
    
    logging.info(f"Found {len(midi_files)} MIDI files in {source_dir}")
    return midi_files

def sample_and_move_files(midi_files, dest_dir, sample_size):
    """Randomly sample MIDI files and move them to the destination directory."""
    if not midi_files:
        logging.error("No MIDI files found to sample from")
        return
    
    # Ensure sample size is not larger than available files
    sample_size = min(sample_size, len(midi_files))
    sampled_files = random.sample(midi_files, sample_size)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Move files
    for file_path in sampled_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)
        
        # Handle filename collisions
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            dest_path = os.path.join(dest_dir, f"{base}_{random.randint(1000, 9999)}{ext}")
        
        shutil.move(file_path, dest_path)
        logging.info(f"Moved: {file_path} -> {dest_path}")
    
    logging.info(f"Successfully moved {len(sampled_files)} MIDI files to {dest_dir}")

def main():
    parser = argparse.ArgumentParser(description="Randomly sample MIDI files and move them to a test folder")
    parser.add_argument("--source", type=str, required=True, help="Source directory containing MIDI files")
    parser.add_argument("--dest", type=str, default="test_midi", help="Destination directory for sampled files")
    parser.add_argument("--count", type=int, default=10, help="Number of files to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    random.seed(args.seed)
    
    # Find and sample files
    midi_files = find_midi_files(args.source)
    if midi_files:
        sample_and_move_files(midi_files, args.dest, args.count)
    else:
        logging.error(f"No MIDI files found in {args.source}")

if __name__ == "__main__":
    main()
