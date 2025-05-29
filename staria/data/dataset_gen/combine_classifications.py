#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import glob
from pathlib import Path

def combine_classifications():
    """
    Combines classification results from multiple folders into a single dataset.
    Creates:
    1. A combined CSV of all predictions
    2. A combined CSV of high confidence predictions
    3. A central folder with all MIDI files
    """
    print("Starting to combine classification results...")
    
    # Base directory for results
    base_results_dir = "./aria-classify-v1"
    
    # Output directories and files
    output_dir = os.path.join(base_results_dir, "combined_results")
    os.makedirs(output_dir, exist_ok=True)
    
    combined_midi_dir = os.path.join(output_dir, "midi_files")
    os.makedirs(combined_midi_dir, exist_ok=True)
    
    combined_predictions_csv = os.path.join(output_dir, "all_predictions.csv")
    combined_high_confidence_csv = os.path.join(output_dir, "high_confidence.csv")
    
    # Find all result folders
    result_folders = [f for f in os.listdir(base_results_dir) 
                     if os.path.isdir(os.path.join(base_results_dir, f)) 
                     and f.startswith("real_results_")]
    
    print(f"Found {len(result_folders)} result folders to process")
    
    # Initialize empty DataFrames for combined results
    all_predictions_df = pd.DataFrame()
    high_confidence_df = pd.DataFrame()
    
    # Process each folder
    for folder in result_folders:
        folder_path = os.path.join(base_results_dir, folder)
        print(f"Processing folder: {folder}")
        
        # Get the dataset name (e.g., 'ko' from 'real_results_ko')
        dataset_name = folder.replace("real_results_", "")
        
        # Process predictions CSV
        predictions_csv = os.path.join(folder_path, "midi_predictions.csv")
        if os.path.exists(predictions_csv):
            try:
                df = pd.read_csv(predictions_csv)
                all_predictions_df = pd.concat([all_predictions_df, df], ignore_index=True)
                print(f"  Added {len(df)} predictions from {folder}")
            except Exception as e:
                print(f"  Error processing predictions from {folder}: {e}")
        else:
            print(f"  No predictions.csv found in {folder}")
        
        # Process high confidence CSV
        high_conf_csv = os.path.join(folder_path, "high_confidence.csv")
        if os.path.exists(high_conf_csv):
            try:
                df = pd.read_csv(high_conf_csv)
                high_confidence_df = pd.concat([high_confidence_df, df], ignore_index=True)
                print(f"  Added {len(df)} high confidence entries from {folder}")
            except Exception as e:
                print(f"  Error processing high confidence from {folder}: {e}")
        else:
            print(f"  No high_confidence.csv found in {folder}")
        
        # Copy MIDI files
        midi_folder = os.path.join(folder_path, "midi_files")
        if os.path.exists(midi_folder):
            midi_files = glob.glob(os.path.join(midi_folder, "*.mid"))
            for midi_file in midi_files:
                # Create a unique filename by prefixing with dataset name
                filename = os.path.basename(midi_file)
                new_filename = f"{dataset_name}_{filename}"
                dest_path = os.path.join(combined_midi_dir, new_filename)
                
                try:
                    shutil.copy2(midi_file, dest_path)
                except Exception as e:
                    print(f"  Error copying {midi_file}: {e}")
            
            print(f"  Copied {len(midi_files)} MIDI files from {folder}")
        else:
            print(f"  No midi_files folder found in {folder}")
    
    # Save combined results
    if not all_predictions_df.empty:
        all_predictions_df.to_csv(combined_predictions_csv, index=False)
        print(f"Saved combined predictions to {combined_predictions_csv}")
    else:
        print("No predictions to combine")
    
    if not high_confidence_df.empty:
        high_confidence_df.to_csv(combined_high_confidence_csv, index=False)
        print(f"Saved combined high confidence entries to {combined_high_confidence_csv}")
    else:
        print("No high confidence entries to combine")
    
    # Count total files
    total_midi_files = len(glob.glob(os.path.join(combined_midi_dir, "*.mid")))
    print(f"Total MIDI files in combined directory: {total_midi_files}")
    
    print("Combination complete!")

if __name__ == "__main__":
    combine_classifications()

