import os
import pickle
import random
import argparse
from typing import Dict, List, Any, Tuple

def load_pkl(file_path: str) -> Any:
    """Load a pickle file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data: Any, file_path: str) -> None:
    """Save data to a pickle file."""
    print(f"Saving data to {file_path}...")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved successfully to {file_path}")

def split_data(data: List[Any], train_ratio: float = 0.9) -> Tuple[List[Any], List[Any]]:
    """Split data into training and validation sets."""
    # Shuffle the data to ensure randomness
    random.shuffle(data)
    
    # Calculate split index
    split_idx = int(len(data) * train_ratio)
    
    # Split the data
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Split data: {len(data)} total items")
    print(f"  - Training set: {len(train_data)} items ({train_ratio*100:.1f}%)")
    print(f"  - Validation set: {len(val_data)} items ({(1-train_ratio)*100:.1f}%)")
    
    return train_data, val_data

def main():
    parser = argparse.ArgumentParser(description="Split a pickle file into training and validation sets")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input pickle file")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the output files")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data to use for training (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_pkl(args.input_file)
    
    # Get base filename without extension
    base_name = os.path.basename(args.input_file)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Split data
    train_data, val_data = split_data(data, args.train_ratio)
    
    # Save split data
    train_output_path = os.path.join(args.output_dir, f"{name_without_ext}_train.pkl")
    val_output_path = os.path.join(args.output_dir, f"{name_without_ext}_val.pkl")
    
    save_pkl(train_data, train_output_path)
    save_pkl(val_data, val_output_path)
    
    print("Data splitting complete!")

if __name__ == "__main__":
    main()
