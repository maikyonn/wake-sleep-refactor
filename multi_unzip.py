#!/usr/bin/env python3
"""
Multi-threaded tar.gz extraction utility.
Uses all available CPU cores to speed up extraction of large archives.
"""

import os
import sys
import tarfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def extract_member(tar_path, output_dir, member):
    """Extract a single member from the archive."""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extract(member, path=output_dir)
        return True, member.name
    except Exception as e:
        return False, f"{member.name}: {str(e)}"

def multi_extract(tar_path, output_dir=None, num_workers=None):
    """
    Extract a tar.gz file using multiple processes.
    
    Args:
        tar_path: Path to the tar.gz file
        output_dir: Directory to extract to (defaults to current directory)
        num_workers: Number of worker processes (defaults to CPU count)
    """
    if not os.path.exists(tar_path):
        logger.error(f"Archive not found: {tar_path}")
        return False
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    logger.info(f"Extracting {tar_path} to {output_dir} using {num_workers} workers")
    
    # First, get the list of members
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
    
    total_size = sum(m.size for m in members)
    logger.info(f"Archive contains {len(members)} files, total size: {total_size / (1024*1024):.2f} MB")
    
    # Create a partial function with the fixed arguments
    extract_fn = partial(extract_member, tar_path, output_dir)
    
    # Use ProcessPoolExecutor to extract in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(extract_fn, members),
            total=len(members),
            desc="Extracting files",
            unit="files"
        ))
    
    # Check for errors
    errors = [msg for success, msg in results if not success]
    if errors:
        logger.error(f"Encountered {len(errors)} errors during extraction")
        for error in errors[:10]:  # Show first 10 errors
            logger.error(error)
        if len(errors) > 10:
            logger.error(f"... and {len(errors) - 10} more errors")
        return False
    
    logger.info(f"Successfully extracted {len(members)} files to {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded tar.gz extraction utility")
    parser.add_argument("archive", help="Path to the tar.gz file to extract")
    parser.add_argument("-o", "--output-dir", help="Directory to extract to (default: current directory)")
    parser.add_argument("-w", "--workers", type=int, help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()
    
    success = multi_extract(args.archive, args.output_dir, args.workers)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
