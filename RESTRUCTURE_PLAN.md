# Repository Restructure Plan

## New Directory Structure

```
staria/                     # Main package
├── __init__.py
├── models/                 # Model architectures
│   ├── __init__.py
│   ├── staria_model.py     # Main Staria encoder-decoder
│   ├── decoder_model.py    # Decoder-only models
│   ├── encoder_model.py    # Encoder-only models
│   └── tokenizer.py        # Music tokenizer
├── data/                   # Data handling
│   ├── __init__.py
│   ├── midi_dataset.py     # MIDI data loading
│   ├── synthetic_gen.py    # Synthetic data generation
│   └── preprocessing.py    # Data preprocessing utilities
├── training/               # Training infrastructure
│   ├── __init__.py
│   ├── trainers.py         # Training modules
│   ├── callbacks.py        # Custom callbacks
│   └── schedulers.py       # Learning rate scheduling
├── evaluation/             # Evaluation frameworks
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics
│   └── human_eval.py       # Human evaluation tools
├── generation/             # Generation utilities
│   ├── __init__.py
│   ├── generators.py       # Generation classes
│   └── post_processing.py  # Post-processing utilities
├── utils/                  # General utilities
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── logging.py          # Logging utilities
│   └── midi_utils.py       # MIDI manipulation tools
└── baselines/              # Baseline implementations
    ├── __init__.py
    └── decoder_only.py     # Decoder-only baseline

scripts/                    # Executable scripts
├── train_staria.py         # Train full Staria model
├── train_decoder_only.py   # Train decoder baseline
├── train_encoder.py        # Train encoder only
├── generate.py             # Generation script
└── evaluate.py             # Evaluation script

experiments/                # Experiment configurations
├── configs/                # YAML configuration files
└── results/                # Experiment results

external/                   # External dependencies
├── aria/                   # Aria tokenizer
├── x_transformers/         # Transformer library
└── ariautils/              # Aria utilities

docs/                       # Documentation
├── api/                    # API documentation
└── tutorials/              # Usage tutorials

tests/                      # Test suite
├── unit/                   # Unit tests
└── integration/            # Integration tests
```

## Migration Strategy

1. **Create new structure** while preserving existing code
2. **Move files systematically** to new locations
3. **Update imports** throughout codebase
4. **Create decoder-only baseline** as part of restructure
5. **Test everything works** after migration
6. **Commit changes** in logical chunks

## Benefits

- **Clear separation of concerns**
- **Easier to find and modify components**
- **Better testing structure**
- **Scalable for future development**
- **Industry-standard organization**