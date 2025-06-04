#!/usr/bin/env python3
"""
Multi-threaded tar.gz compression and extraction utility.
Uses all available CPU cores to speed up compression/extraction of large archives.
"""

import os
import sys
import tarfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import threading

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

def add_file_to_tar(tar_lock, tar_path, file_path, arcname):
    """Add a single file to the archive (thread-safe)."""
    try:
        with tar_lock:
            with tarfile.open(tar_path, 'a:gz') as tar:
                tar.add(file_path, arcname=arcname)
        return True, arcname
    except Exception as e:
        return False, f"{arcname}: {str(e)}"

def collect_files(source_path):
    """Collect all files to be compressed."""
    files = []
    if os.path.isfile(source_path):
        files.append((source_path, os.path.basename(source_path)))
    elif os.path.isdir(source_path):
        source_path = Path(source_path)
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                # Calculate relative path for archive
                arcname = str(file_path.relative_to(source_path.parent))
                files.append((str(file_path), arcname))
    return files

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

def multi_compress(source_path, tar_path, num_workers=None):
    """
    Compress files/directories into a tar.gz file using multiple threads.
    
    Args:
        source_path: Path to file or directory to compress
        tar_path: Path for the output tar.gz file
        num_workers: Number of worker threads (defaults to CPU count)
    """
    if not os.path.exists(source_path):
        logger.error(f"Source not found: {source_path}")
        return False
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # Collect all files to compress
    files = collect_files(source_path)
    if not files:
        logger.error(f"No files found to compress in: {source_path}")
        return False
    
    total_size = sum(os.path.getsize(f[0]) for f in files)
    logger.info(f"Compressing {len(files)} files, total size: {total_size / (1024*1024):.2f} MB")
    logger.info(f"Creating archive: {tar_path} using {num_workers} workers")
    
    # Create empty archive first
    with tarfile.open(tar_path, 'w:gz') as tar:
        pass
    
    # Use a lock to ensure thread-safe access to the tar file
    tar_lock = threading.Lock()
    
    # Create a partial function with the fixed arguments
    add_fn = partial(add_file_to_tar, tar_lock, tar_path)
    
    # Use ThreadPoolExecutor (not ProcessPoolExecutor) because we need shared file access
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda x: add_fn(x[0], x[1]), files),
            total=len(files),
            desc="Compressing files",
            unit="files"
        ))
    
    # Check for errors
    errors = [msg for success, msg in results if not success]
    if errors:
        logger.error(f"Encountered {len(errors)} errors during compression")
        for error in errors[:10]:  # Show first 10 errors
            logger.error(error)
        if len(errors) > 10:
            logger.error(f"... and {len(errors) - 10} more errors")
        return False
    
    final_size = os.path.getsize(tar_path)
    compression_ratio = (1 - final_size / total_size) * 100 if total_size > 0 else 0
    logger.info(f"Successfully compressed {len(files)} files to {tar_path}")
    logger.info(f"Archive size: {final_size / (1024*1024):.2f} MB, compression ratio: {compression_ratio:.1f}%")
    return True

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded tar.gz compression and extraction utility")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract a tar.gz archive')
    extract_parser.add_argument("archive", help="Path to the tar.gz file to extract")
    extract_parser.add_argument("-o", "--output-dir", help="Directory to extract to (default: current directory)")
    extract_parser.add_argument("-w", "--workers", type=int, help="Number of worker processes (default: CPU count)")
    
    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Create a tar.gz archive')
    compress_parser.add_argument("source", help="Path to file or directory to compress")
    compress_parser.add_argument("archive", help="Path for the output tar.gz file")
    compress_parser.add_argument("-w", "--workers", type=int, help="Number of worker threads (default: CPU count)")
    
    # Legacy support: if no subcommand is given, assume extract
    if len(sys.argv) > 1 and sys.argv[1] not in ['extract', 'compress', '-h', '--help']:
        # Insert 'extract' as the first argument
        sys.argv.insert(1, 'extract')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        success = multi_extract(args.archive, args.output_dir, args.workers)
    elif args.command == 'compress':
        success = multi_compress(args.source, args.archive, args.workers)
    else:
        parser.print_help()
        sys.exit(1)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
