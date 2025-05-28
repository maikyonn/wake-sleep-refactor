# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Staria model - a wake-sleep refactored MIDI music generation system using PyTorch Lightning. The project implements an encoder-decoder transformer architecture for generating structured MIDI music with style conditioning.

## Architecture

### Core Components
- **StariaModule.py**: Main PyTorch Lightning module containing the Staria model (encoder-decoder architecture)
- **MidiDataModule.py**: Data loading and preprocessing for MIDI files with support for synthetic and real data
- **StariaTokenizer.py**: Music tokenizer with style tokens and special structural tokens (sections A/B/C/D)
- **AriaTransformerModel.py**: Transformer model implementations using x-transformers library

### Key Directories
- `src/`: Core model implementations and utilities
- `src/aria/`: Aria model components and utilities
- `src/x_transformers/`: Custom transformer implementations
- `base-weights/`: Pre-trained model checkpoints
- `datasets/`: Training data location
- `cache/`: Cached dataset paths (pickle files)
- `cc/`: Comma Controls Challenge - separate control system project

### Training Pipeline
The project supports multiple training phases:
1. **Encoder training** (`train_encoder.py`) - Contrastive learning phase
2. **Decoder training** (`train_decoder.py`) - Autoregressive generation
3. **Joint training** (`train_staria.py`) - Full encoder-decoder training

## Common Commands

### Environment Setup
```bash
# Activate conda environment (adjust path as needed)
source activate /pool001/spangher/wake-sleep-refactor/wake-sleep-env
```

### Training Commands
```bash
# Train Staria model with DDP on SLURM
sbatch train_2_staria.sh

# Train encoder only
python train_encoder.py --data_dir cache/dataset_paths_*.pkl --epochs 10

# Train decoder only  
python train_decoder.py --data_dir cache/dataset_paths_*.pkl --epochs 10

# Train full model locally
python train_staria.py --data_dir cache/dataset_paths_*.pkl --val_data_dir cache/dataset_paths_*_val.pkl
```

### Generation
```bash
# Generate MIDI samples
python fabric_generate.py --checkpoint_path checkpoints/final_model.ckpt
```

### Data Processing
```bash
# Create dataset splits
python create_pkl_split.py

# Preprocess MIDI files
python src/preprocess.py --input_dir datasets/ --output_dir cache/
```

## Data Configuration

The project uses pickle files containing dataset paths for efficient loading:
- Training data: `cache/dataset_paths_*_train.pkl`
- Validation data: `cache/dataset_paths_*_val.pkl`

Data modes:
- `synthetic`: Generated MIDI data
- `real`: Actual MIDI recordings

Tasks:
- `generative`: Standard autoregressive generation
- `contrastive`: Encoder training with contrastive loss
- `classification`: Style/form classification

## Model Configuration

Key hyperparameters:
- Max sequence length: 4096 tokens
- Encoder: 6 layers, 8 heads, 1536 dim
- Decoder: 16 layers, 24 heads, 1536 dim
- Supports snippet mode for shorter sequences (256 tokens)

## SLURM Integration

The project is designed for SLURM clusters with:
- Multi-node DDP training support
- GPU memory monitoring
- Checkpoint saving to `checkpoints/` directory
- Logging to `slurm_logs/`

Key SLURM scripts:
- `train_2_staria.sh`: Main Staria training
- `train_1_base_enc.sh`: Base encoder training
- `train_1_base_dec.sh`: Base decoder training

## Special Tokens

The tokenizer includes structural tokens:
- Section markers: `A_SECTION_TOKEN`, `B_SECTION_TOKEN`, etc.
- Prompt delimiters: `PROMPT_START_TOKEN`, `PROMPT_END_TOKEN`
- Style conditioning tokens in `FORM_LABEL_MAP`