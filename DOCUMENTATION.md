# Staria: Hierarchical Music Generation System
## Complete Documentation and User Guide

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Installation and Setup](#installation-and-setup)
5. [Training Commands](#training-commands)
6. [Generation Commands](#generation-commands)
7. [Smart Checkpointing System](#smart-checkpointing-system)
8. [Model Components](#model-components)
9. [Data Handling](#data-handling)
10. [Evaluation Framework](#evaluation-framework)
11. [Research Contributions](#research-contributions)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Staria** is a hierarchical music generation system that combines encoder-decoder architectures with wake-sleep inspired training for structured MIDI music generation. The system learns compressed musical representations through snippet-based encoding and generates coherent, long-form music through conditioned decoding.

### Key Features
- **Hierarchical Architecture**: Encoder-decoder design with snippet-based conditioning
- **Wake-Sleep Training**: Inspired training methodology for musical understanding
- **Structural Awareness**: Explicit modeling of musical sections (A, B, C, D forms)
- **Style Conditioning**: Generation with specific stylistic characteristics
- **Smart Checkpointing**: Automatic checkpoint management with SLURM job chaining
- **Baseline Comparisons**: Decoder-only baseline for systematic evaluation

### Novel Contributions
- **Snippet-based conditioning**: Efficient method for long-form generation
- **Form-aware synthesis**: Training data with balanced musical forms
- **Staged training protocol**: Effective training strategy for complex musical models
- **Multi-scale musical modeling**: Architecture capturing musical hierarchy

---

## Architecture

### Core Design
```
Musical Snippet → Encoder (6L, 8H, 1536D) → Compressed Representation
                                               ↓
Decoder Prompt → Decoder (16L, 24H, 1536D) → Full Generation
```

### Key Components
- **Encoder**: 6 layers, 8 heads, 1536 dimensions - processes 256-token snippets
- **Decoder**: 16 layers, 24 heads, 1536 dimensions - generates full sequences (4096 tokens)
- **Cross-attention**: Connects encoder representations to decoder generation
- **Style Tokenizer**: 17,742 vocabulary size with structural and style tokens

### Training Methodology
1. **Stage A**: Cross-attention and adapter training
2. **Stage B**: Progressive decoder unfreezing
3. **Stage C**: Full system fine-tuning with encoder unfreezing

---

## Repository Structure

```
staria/                          # Main Python package
├── models/                      # Model architectures
│   ├── tokenizer.py            # MusicTokenizerWithStyle
│   ├── staria_model.py         # StariaModule (encoder-decoder)
│   ├── legacy_models.py        # OldStariaModule components
│   ├── aria_transformer.py     # AriaTransformerModel
│   ├── generator.py            # GeneratorModule
│   └── snippet_encoder.py      # SnippetModule
├── baselines/                   # Baseline implementations
│   └── decoder_only.py         # DecoderOnlyBaseline (513M params)
├── data/                        # Data handling and preprocessing
│   ├── midi_dataset.py         # MidiDataModule
│   ├── preprocessing.py        # MIDI preprocessing utilities
│   ├── path_preprocessing.py   # Dataset path handling
│   └── create_pkl_split.py     # Dataset splitting
├── generation/                  # Music generation utilities
│   ├── fabric_generator.py     # Combined generation pipeline
│   └── fabric_decoder.py       # Decoder generation
├── evaluation/                  # Evaluation frameworks
│   ├── classifier_metrics.py   # Style classification evaluation
│   └── data_statistics.py      # Dataset statistics
├── training/                    # Training infrastructure
│   ├── checkpoint_callbacks.py # Smart checkpointing callbacks
│   └── __init__.py             # Training utilities
└── utils/                       # General utilities
    ├── checkpoint_manager.py   # Checkpoint management
    ├── general_utils.py        # General utilities
    └── music_utils.py          # Music-specific utilities

scripts/                         # Executable training/generation scripts
├── train_staria.py             # Train full Staria model
├── train_decoder_only.py      # Train decoder baseline
├── train_encoder.py            # Train encoder only
├── train_decoder.py            # Train decoder only
├── generate_decoder_only.py   # Generate with decoder baseline
├── train_staria_auto_resume.sh # Auto-resume Staria training
├── train_decoder_auto_resume.sh # Auto-resume decoder training
└── launch_training_chain.py   # Training chain launcher

external/                        # External dependencies
├── ariautils/                   # Aria tokenizer utilities
├── x_transformers/             # Transformer implementations
├── aria_original/              # Original Aria implementation
└── aria_generative/            # Aria generative models

base-weights/                    # Pre-trained model checkpoints
checkpoints/                     # Training checkpoints
cache/                          # Cached dataset paths
datasets/                       # Training data location
```

---

## Installation and Setup

### Environment Activation
```bash
# Activate conda environment
source activate /pool001/spangher/wake-sleep-refactor/wake-sleep-env

# Verify installation
which python
cd /pool001/spangher/wake-sleep-refactor
```

### Required Dependencies
- PyTorch Lightning
- Transformers
- PyTorch with CUDA support
- Aria utilities
- MIDI processing libraries

---

## Training Commands

### Full Staria Model Training
```bash
# Local training
python scripts/train_staria.py \
    --data_dir cache/dataset_paths_*_train.pkl \
    --val_data_dir cache/dataset_paths_*_val.pkl \
    --epochs 50 \
    --batch_size 4

# SLURM distributed training
sbatch scripts/train_staria_auto_resume.sh
```

### Decoder-Only Baseline Training
```bash
# Local training
python scripts/train_decoder_only.py \
    --data_dir datasets/midi \
    --batch_size 4 \
    --epochs 50 \
    --max_len 4096 \
    --snippet_length 256 \
    --num_snippets 2

# SLURM distributed training
sbatch scripts/train_decoder_auto_resume.sh
```

### Component Training
```bash
# Encoder only (contrastive pre-training)
python scripts/train_encoder.py \
    --data_dir cache/dataset_paths_*.pkl \
    --epochs 10

# Decoder only (autoregressive)
python scripts/train_decoder.py \
    --data_dir cache/dataset_paths_*.pkl \
    --epochs 10
```

### Training Chain Management
```bash
# Launch training chains with auto-resume
python scripts/launch_training_chain.py staria
python scripts/launch_training_chain.py decoder

# Custom job names
python scripts/launch_training_chain.py staria --job-name my_experiment

# Monitor jobs
squeue -u $USER
tail -f slurm_logs/staria_auto_chain-*.out

# Cancel job chains
scancel --name staria_auto_chain
```

---

## Generation Commands

### Staria Model Generation
```bash
# Generate with Staria model
python staria/generation/fabric_generator.py \
    --checkpoint_path checkpoints/staria/final_model.ckpt \
    --num_samples 5 \
    --output_dir generated_samples/
```

### Decoder Baseline Generation
```bash
# Generate with decoder baseline
python scripts/generate_decoder_only.py \
    --checkpoint checkpoints/decoder_baseline/final_model.ckpt \
    --num_samples 5 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9
```

### Generation Parameters
- `--temperature`: Controls randomness (0.1-2.0)
- `--top_k`: Top-k sampling parameter
- `--top_p`: Nucleus sampling parameter
- `--max_length`: Maximum sequence length
- `--num_samples`: Number of samples to generate

---

## Checkpointing System

### Overview
Standard PyTorch Lightning checkpointing with top-2 retention and SLURM job chaining for continuous training.

### Key Features
- **Top-2 Retention**: Uses PyTorch Lightning ModelCheckpoint with save_top_k=2
- **Automatic Resume**: Resumes from `last.ckpt` checkpoint
- **Job Chaining**: SLURM scripts automatically queue next job when current completes
- **Standard Integration**: Uses built-in PyTorch Lightning callbacks

### Components
- **ModelCheckpoint**: Standard PyTorch Lightning checkpoint callback
- **SLURM Scripts**: Auto-resume functionality looking for `last.ckpt`
- **Training Launcher**: Convenient job submission utility

### Usage
```python
# Integrate in training scripts
from staria.training import setup_checkpointing

callbacks = setup_checkpointing(
    checkpoint_dir="checkpoints/my_model",
    monitor="val_loss",
    save_top_k=2
)

trainer = pl.Trainer(callbacks=callbacks, ...)
```

### Checkpoint Directory Structure
```
checkpoints/
├── staria/
│   ├── last.ckpt                    # Latest checkpoint (for resume)
│   ├── epoch-015-val_loss-2.3456.ckpt  # Best checkpoint 1
│   ├── epoch-023-val_loss-2.2891.ckpt  # Best checkpoint 2
│   └── checkpoint_history.json     # Metadata tracking
├── decoder_baseline/
│   └── ...
└── encoder/
    └── ...
```

---

## Model Components

### StariaModule (`staria/models/staria_model.py`)
**Inputs**: Snippet tokens, decoder prompts, style labels
**Outputs**: Generated music sequences, attention weights
**Key Parameters**: encoder_layers=6, decoder_layers=16, d_model=1536

### DecoderOnlyBaseline (`staria/baselines/decoder_only.py`)
**Inputs**: Prompted sequences with snippet context
**Outputs**: Autoregressive music generation
**Key Parameters**: 513M parameters, 16 layers, 24 heads
**Training Format**: `<PROMPT_START> <A_SECTION> [snippet] <PROMPT_END> [sequence]`

### MusicTokenizerWithStyle (`staria/models/tokenizer.py`)
**Inputs**: MIDI files, style labels, section markers
**Outputs**: Tokenized sequences with structural tokens
**Vocabulary**: 17,742 tokens including special structural markers

### MidiDataModule (`staria/data/midi_dataset.py`)
**Inputs**: Dataset paths (pickle files), data modes (synthetic/real)
**Outputs**: Batched training data with snippets and full sequences
**Supports**: Contrastive pairs, generative sequences, classification labels

---

## Data Handling

### Dataset Configuration
```bash
# Training data paths
cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_train.pkl
cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_val.pkl
```

### Data Modes
- **synthetic**: Generated MIDI data with balanced forms
- **real**: Actual MIDI recordings

### Tasks
- **generative**: Standard autoregressive generation
- **contrastive**: Encoder training with NT-Xent loss
- **classification**: Style/form classification

### Preprocessing
```bash
# Create dataset splits
python staria/data/create_pkl_split.py

# Preprocess MIDI files
python staria/data/preprocessing.py \
    --input_dir datasets/ \
    --output_dir cache/
```

---

## Evaluation Framework

### Style Classification (`staria/evaluation/classifier_metrics.py`)
**Inputs**: Generated samples, style labels
**Outputs**: Classification accuracy, confusion matrices
**Metrics**: Precision, recall, F1-score per style class

### Data Statistics (`staria/evaluation/data_statistics.py`)
**Inputs**: MIDI datasets
**Outputs**: Distribution analyses, form statistics
**Metrics**: Note distributions, rhythm patterns, form balance

### Evaluation Commands
```bash
# Evaluate style classification
python staria/evaluation/classifier_metrics.py \
    --generated_dir generated_samples/ \
    --ground_truth_dir datasets/validation/

# Generate dataset statistics
python staria/evaluation/data_statistics.py \
    --data_dir datasets/
```

---

## Research Contributions

### Technical Innovations
1. **Hierarchical Music Generation**: Novel encoder-decoder architecture for structured music
2. **Wake-Sleep Musical Learning**: Adaptation of wake-sleep algorithms to music domain
3. **Snippet-Based Conditioning**: Efficient method for long-form generation
4. **Staged Training Protocol**: Effective training strategy for complex musical models

### Methodological Advances
1. **Form-Aware Synthesis**: Methods for creating balanced, structure-aware training data
2. **Style-Aware Tokenization**: Advanced tokenization preserving musical structure
3. **Multi-Scale Musical Modeling**: Architecture capturing musical hierarchy
4. **Comprehensive Evaluation**: Framework for objective and subjective music assessment

### System Capabilities
- **Long-form generation**: Coherent music up to 4096 tokens
- **Style conditioning**: Generate music with specific characteristics
- **Structural awareness**: Explicit A/B/C/D section modeling
- **Efficient training**: Hierarchical approach reduces computational requirements

---

## Troubleshooting

### Training Issues
```bash
# Check GPU memory usage
nvidia-smi

# Monitor training logs
tail -f slurm_logs/*.out

# Check checkpoint status
ls -la checkpoints/staria/
```

### Generation Issues
```bash
# Verify checkpoint loading
python -c "
import torch
ckpt = torch.load('checkpoints/staria/last.ckpt', map_location='cpu')
print('Checkpoint keys:', ckpt.keys())
"

# Test tokenizer
python -c "
from staria.models.tokenizer import MusicTokenizerWithStyle
tokenizer = MusicTokenizerWithStyle()
print('Vocab size:', tokenizer.vocab_size())
"
```

### SLURM Job Issues
```bash
# Check job status
squeue -j <job_id>

# View job details
scontrol show job <job_id>

# Check node availability
sinfo -p sched_mit_psfc_gpu_r8
```

### Checkpoint Issues
```bash
# Manual checkpoint cleanup
python -c "
from staria.utils.checkpoint_manager import CheckpointManager
manager = CheckpointManager('checkpoints/staria')
manager._cleanup_old_checkpoints()
"

# Check checkpoint history
cat checkpoints/staria/checkpoint_history.json
```

### Common Solutions
1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Import errors**: Verify environment activation and package structure
3. **Data loading errors**: Check pickle file paths and permissions
4. **Training stalling**: Monitor learning rates and gradient norms
5. **Generation quality**: Adjust temperature and sampling parameters

---

## Special Tokens and Configuration

### Structural Tokens
- Section markers: `A_SECTION_TOKEN`, `B_SECTION_TOKEN`, `C_SECTION_TOKEN`, `D_SECTION_TOKEN`
- Prompt delimiters: `PROMPT_START_TOKEN`, `PROMPT_END_TOKEN`
- Piano prefix: Tokens `6, 0` for instrument specification

### Model Hyperparameters
- **Max sequence length**: 4096 tokens
- **Snippet length**: 256 tokens (configurable)
- **Encoder**: 6 layers, 8 heads, 1536 dimensions
- **Decoder**: 16 layers, 24 heads, 1536 dimensions
- **Vocabulary size**: 17,742 tokens

### SLURM Configuration
- **Nodes**: 4 nodes for Staria, 2 for decoder baseline
- **GPUs per node**: 4
- **Memory**: 256GB for Staria, 128GB for decoder
- **Time limit**: 6 hours per job
- **Queue**: `sched_mit_psfc_gpu_r8`

---

This documentation provides comprehensive guidance for using the Staria music generation system. For additional support or advanced configuration, refer to the individual component documentation within the codebase.