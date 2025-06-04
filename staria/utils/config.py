"""
Configuration Management System

Provides utilities for loading and managing JSON configuration files
for training experiments.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import argparse


@dataclass
class DataConfig:
    """Data configuration."""
    train_index: str = "cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_train.pkl"
    val_index: str = "cache/dataset_paths_synthetic_aria-midi-v1-pruned-ext-200k-struct_limitNone_7b76cfca_val.pkl"
    batch_size: int = 2
    num_workers: int = 8
    max_len: int = 4096
    snippet_length: int = 256  # For decoder baseline
    num_snippets: int = 2      # For decoder baseline


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Encoder settings
    encoder_dim: int = 1536
    encoder_depth: int = 6
    encoder_heads: int = 8
    
    # Decoder settings
    decoder_dim: int = 1536
    decoder_depth: int = 16
    decoder_heads: int = 24
    
    # Common settings
    max_len: int = 4096
    vocab_size: Optional[int] = None  # Determined by tokenizer
    
    # Model-specific settings
    model_type: str = "staria"  # staria, decoder_only, classifier
    
    # Classifier specific
    n_classes: int = 4
    dropout_rate: float = 0.1
    hidden: int = 1024
    layers: int = 12
    heads: int = 16
    

@dataclass
class TrainingConfig:
    """Training configuration."""
    lr: float = 3e-4
    epochs: int = 100
    accumulate_grad_batches: int = 4
    gradient_clip_val: float = 1.0
    
    # Hardware
    gpus: int = -1
    num_nodes: int = 1
    strategy: str = "ddp"
    precision: str = "bf16-mixed"
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    check_val_every_n_epoch: int = 1
    enable_progress_bar: bool = True
    

@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""
    project_name: str = "staria"
    run_name: Optional[str] = None
    log_dir: str = "logs"
    use_wandb: bool = True
    

@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 2
    monitor: str = "val_loss"
    mode: str = "min"
    save_last: bool = True
    resume_from: Optional[str] = None
    

@dataclass
class Config:
    """Complete configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Metadata
    experiment_name: str = "default"
    description: str = ""
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "Config":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create sub-configs
        config = cls(
            experiment_name=data.get("experiment_name", "default"),
            description=data.get("description", "")
        )
        
        # Load sub-configs
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "checkpoint" in data:
            config.checkpoint = CheckpointConfig(**data["checkpoint"])
            
        return config
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        data = {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "logging": asdict(self.logging),
            "checkpoint": asdict(self.checkpoint)
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration from command line arguments."""
        # Update data config
        for key in ["train_index", "val_index", "batch_size", "num_workers", "max_len"]:
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(self.data, key, getattr(args, key))
        
        # Update model config
        for key in ["encoder_dim", "encoder_depth", "encoder_heads", 
                    "decoder_dim", "decoder_depth", "decoder_heads"]:
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(self.model, key, getattr(args, key))
        
        # Update training config
        for key in ["lr", "epochs", "accumulate_grad_batches", "gpus", 
                    "num_nodes", "strategy", "precision"]:
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(self.training, key, getattr(args, key))
        
        # Update logging config
        for key in ["project_name", "run_name", "log_dir"]:
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(self.logging, key, getattr(args, key))
        
        # Update checkpoint config
        for key in ["checkpoint_dir", "resume_from"]:
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(self.checkpoint, key, getattr(args, key))
    
    def get_checkpoint_dir(self) -> str:
        """Get checkpoint directory for this experiment."""
        base_dir = self.checkpoint.checkpoint_dir
        model_dir = self.model.model_type
        return os.path.join(base_dir, model_dir, self.experiment_name)
    
    def get_log_dir(self) -> str:
        """Get log directory for this experiment."""
        return os.path.join(self.logging.log_dir, self.model.model_type, self.experiment_name)
    
    def get_run_name(self) -> str:
        """Get run name for experiment tracking."""
        if self.logging.run_name:
            return self.logging.run_name
        return f"{self.experiment_name}_lr{self.training.lr}_bs{self.data.batch_size}"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Path to JSON config file. If None, returns default config.
        
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        return Config.from_json(config_path)
    return Config()


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Config object to save
        save_path: Path to save JSON file
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    config.to_json(save_path)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add configuration arguments to argument parser.
    
    Args:
        parser: Argument parser to add arguments to
        
    Returns:
        Updated argument parser
    """
    # Config file argument
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON configuration file")
    
    # Allow overriding specific values
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name (overrides config)")
    parser.add_argument("--save_config", type=str, default=None,
                       help="Save configuration to this path")
    
    return parser