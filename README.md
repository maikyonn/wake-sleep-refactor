# Staria: Hierarchical Music Generation via Wake-Sleep Learning

A framework for structured music generation using hierarchical encoder-decoder architectures with wake-sleep inspired training methodology.

## 🎵 Overview

Staria implements a novel approach to long-form music generation that combines:
- **Hierarchical Architecture**: Encoder-decoder design for snippet-based conditioning
- **Wake-Sleep Training**: Inspired by wake-sleep algorithms for efficient learning
- **Structured Music Understanding**: Explicit modeling of musical forms (A/B/C/D sections)
- **Decoder-Only Baseline**: Pure transformer baseline for comparison

## 🏗️ Architecture

### Full Staria Model
```
Snippet Input → Encoder (6L, 8H, 1536D) → Compressed Representation
                                             ↓
Decoder Prompt → Decoder (16L, 24H, 1536D) → Full Generation
```

### Decoder-Only Baseline
```
Prompt + MIDI Sequence → Decoder (16L, 24H, 1536D) → Continued Generation
```

## 📁 Repository Structure

```
staria/                     # Main package
├── models/                 # Model architectures
│   ├── tokenizer.py        # Music tokenizer with style tokens
│   ├── staria_model.py     # Full encoder-decoder model
│   └── decoder_model.py    # Decoder-only models
├── baselines/              # Baseline implementations
│   └── decoder_only.py     # Decoder-only baseline
├── data/                   # Data handling
├── training/               # Training infrastructure
├── generation/             # Generation utilities
└── utils/                  # General utilities

scripts/                    # Executable scripts
├── train_decoder_only.py   # Train decoder baseline
├── generate_decoder_only.py # Generate with decoder baseline
├── train_staria.py         # Train full Staria model
└── generate.py             # Generate with Staria

external/                   # External dependencies
├── ariautils/              # Aria tokenizer utilities
└── x_transformers/         # Transformer implementations

experiments/                # Experiment configurations
└── configs/                # YAML configuration files
```

## 🚀 Quick Start

### 1. Training Decoder-Only Baseline

The decoder-only baseline trains on MIDI sequences with snippet prompts:

```bash
# Train decoder-only baseline
python scripts/train_decoder_only.py \
    --data_dir datasets/midi \
    --batch_size 4 \
    --epochs 50 \
    --max_len 4096 \
    --snippet_length 256 \
    --num_snippets 2
```

**Training Format:**
```
<PROMPT_START> <A_SECTION> [snippet_A] <B_SECTION> [snippet_B] <PROMPT_END> [full_midi_sequence]
```

### 2. Generation

Generate music with the trained decoder-only model:

```bash
# Generate samples
python scripts/generate_decoder_only.py \
    --checkpoint checkpoints/decoder_baseline/final_model.ckpt \
    --output_dir generated_samples \
    --num_samples 5 \
    --temperature 0.8
```

### 3. Training Full Staria Model (Hierarchical)

```bash
# Train encoder first (contrastive pre-training)
python scripts/train_encoder.py \
    --data_dir datasets/midi \
    --epochs 20

# Train full Staria model
python scripts/train_staria.py \
    --data_dir datasets/midi \
    --encoder_checkpoint checkpoints/encoder/best.ckpt \
    --epochs 50
```

## 📊 Model Comparison

| Model | Architecture | Training | Use Case |
|-------|-------------|----------|----------|
| **Decoder-Only** | Single decoder transformer | Prompted sequences | Baseline comparison |
| **Staria** | Encoder + Decoder | Wake-sleep training | Hierarchical generation |

## 📈 Key Features

### Decoder-Only Baseline
- ✅ **Simple Architecture**: Single transformer decoder
- ✅ **Prompt Conditioning**: Trains on snippet-prompted sequences  
- ✅ **Structured Generation**: Learns musical forms through prompts
- ✅ **Fast Training**: Direct autoregressive training

### Full Staria Model
- ✅ **Hierarchical Learning**: Separate encoding and generation phases
- ✅ **Snippet-Based Conditioning**: Efficient long-form generation
- ✅ **Musical Structure Awareness**: Explicit section modeling
- ✅ **Wake-Sleep Training**: Staged training protocol

## 🔧 Configuration

### Decoder-Only Hyperparameters
```python
{
    "dim": 1536,           # Model dimension
    "depth": 16,           # Number of layers  
    "heads": 24,           # Attention heads
    "max_len": 4096,       # Maximum sequence length
    "snippet_length": 256, # Length of snippet prompts
    "num_snippets": 2,     # Number of snippets per prompt
    "lr": 3e-4,           # Learning rate
    "temperature": 0.8     # Generation temperature
}
```

## 📚 Data Format

### MIDI Processing
- **Tokenization**: Uses Aria tokenizer with musical understanding
- **Section Labels**: A/B/C/D musical form annotations
- **Snippet Extraction**: Extracts representative segments for prompts
- **Timing Preservation**: Maintains proper MIDI timing relationships

### Training Data
- **Synthetic Data**: Form-aware generated training examples
- **Real MIDI**: Preprocessed MIDI files with structure annotations
- **Prompted Sequences**: Snippet prompts + full MIDI for decoder training

## 🎯 Research Contributions

1. **Hierarchical Music Generation**: Novel encoder-decoder approach for structured music
2. **Wake-Sleep Musical Learning**: Adaptation of wake-sleep algorithms to music domain  
3. **Snippet-Based Conditioning**: Efficient method for long-form generation
4. **Decoder-Only Baseline**: Strong baseline for structured music generation

## 📖 Documentation

- [`research.md`](research.md) - Comprehensive research proposal and roadmap
- [`CLAUDE.md`](CLAUDE.md) - Development guidance for Claude Code
- [`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) - Repository organization details

## 🤝 Contributing

The repository is organized for easy development:

1. **Models**: Add new architectures in `staria/models/`
2. **Baselines**: Implement comparisons in `staria/baselines/`
3. **Training**: Extend training logic in `staria/training/`
4. **Scripts**: Add executable scripts in `scripts/`

## 📄 License

[Add your license information here]

## 🔬 Citation

```bibtex
@misc{staria2024,
    title={Staria: Hierarchical Music Generation via Wake-Sleep Learning},
    author={Your Name},
    year={2024},
    url={https://github.com/yourusername/staria}
}
```