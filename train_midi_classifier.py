import os
import datetime
import logging
import csv
import random
from pathlib import Path
from argparse import ArgumentParser

overall_logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Accuracy
from transformers import BertConfig, BertModel, get_cosine_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger

from src.StariaTokenizer import MusicTokenizerWithStyle, IGNORE_LABEL_IDX
from src.MidiDataModule import MidiDataModule, DataCfg, AugmentCfg
from src.utils_new import filter_significant_styles, extract_style_change_timestamps, condense_style_sequence

from x_transformers import TransformerWrapper, Encoder

torch.set_float32_matmul_precision('medium')
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    pl.seed_everything(seed)


class GPUMemoryCallback(Callback):
    """Log GPU memory usage after each batch."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_reserved() / 2**30
            pl_module.log('gpu_memory_train_GB', used_gb, on_step=True, prog_bar=True)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_reserved() / 2**30
            pl_module.log('gpu_memory_val_GB', used_gb, on_step=False, on_epoch=True, prog_bar=True)


class DatasetProgressCallback(Callback):
    """Callback to manage progressive dataset expansion."""
    
    def __init__(self, csv_tracker_path: str, data_module: MidiDataModule, confidence_threshold: float = 0.95):
        self.csv_tracker_path = csv_tracker_path
        self.data_module = data_module
        self.confidence_threshold = confidence_threshold
        self.evaluation_epochs = [5, 10, 15, 20, 25]  # Re-evaluate every 5 epochs
        
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        
        if current_epoch in self.evaluation_epochs:
            overall_logger.info(f"Epoch {current_epoch}: Re-evaluating remaining files for training set expansion")
            self._evaluate_and_expand_training_set(trainer, pl_module)
            
    def _evaluate_and_expand_training_set(self, trainer, pl_module):
        """Evaluate remaining files and add confident predictions to training set."""
        # Load current CSV state
        csv_data = self._load_csv_tracker()
        
        # Get files marked as 'remaining'
        remaining_files = [row for row in csv_data if row['status'] == 'remaining']
        
        if not remaining_files:
            overall_logger.info("No remaining files to evaluate")
            return
            
        overall_logger.info(f"Evaluating {len(remaining_files)} remaining files")
        
        # Create temporary dataset for evaluation
        temp_records = []
        for row in remaining_files:
            temp_records.append({
                'midi_file_path': row['midi_file_path'],
                'style_file_path': row['style_file_path']
            })
        
        # Create temporary pkl file for evaluation
        import pickle
        temp_pkl_path = self.csv_tracker_path.replace('.csv', '_temp_eval.pkl')
        temp_data = {
            'metadata': {'purpose': 'temporary_evaluation'},
            'data_records': temp_records
        }
        
        with open(temp_pkl_path, 'wb') as f:
            pickle.dump(temp_data, f)
        
        # Create evaluation dataset
        eval_cfg = DataCfg(
            data_dir=temp_pkl_path,
            mode='synthetic',
            task='classification',
            max_len=self.data_module.cfg.max_len,
            shuffle_records=False,
            skip_long_after_tokenization=True
        )
        
        from src.MidiDataModule import OnDemandMidiDataset, midi_collate_mapstyle
        eval_dataset = OnDemandMidiDataset(eval_cfg, self.data_module.tokenizer, temp_pkl_path)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.data_module.batch_size,
            collate_fn=lambda batch: midi_collate_mapstyle(batch, self.data_module.tokenizer),
            num_workers=0  # Use single worker for evaluation
        )
        
        # Evaluate files
        pl_module.eval()
        confident_files = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                if batch is None:
                    continue
                    
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                logits = pl_module(batch['input_ids'], batch['attention_mask'])
                probs = F.softmax(logits, dim=-1)
                max_probs, predictions = torch.max(probs, dim=-1)
                
                # Check confidence for each sequence in batch
                for seq_idx in range(logits.size(0)):
                    seq_max_probs = max_probs[seq_idx]
                    seq_predictions = predictions[seq_idx]
                    
                    # Calculate average confidence for non-padded tokens
                    mask = batch['attention_mask'][seq_idx].bool()
                    if mask.sum() > 0:
                        avg_confidence = seq_max_probs[mask].mean().item()
                        
                        if avg_confidence >= self.confidence_threshold:
                            file_idx = batch_idx * self.data_module.batch_size + seq_idx
                            if file_idx < len(remaining_files):
                                confident_files.append(remaining_files[file_idx])
        
        # Clean up temporary file
        os.remove(temp_pkl_path)
        
        # Update CSV tracker
        if confident_files:
            overall_logger.info(f"Adding {len(confident_files)} confident files to training set")
            self._update_csv_status(confident_files, 'training')
            
            # Recreate training dataset with expanded data
            self._recreate_training_dataset()
        else:
            overall_logger.info("No files met confidence threshold for training set expansion")
    
    def _load_csv_tracker(self):
        """Load CSV tracker data."""
        csv_data = []
        with open(self.csv_tracker_path, 'r') as f:
            reader = csv.DictReader(f)
            csv_data = list(reader)
        return csv_data
    
    def _update_csv_status(self, files_to_update, new_status):
        """Update status of specific files in CSV tracker."""
        csv_data = self._load_csv_tracker()
        
        # Create lookup for faster updates
        files_to_update_paths = {row['midi_file_path'] for row in files_to_update}
        
        # Update status
        for row in csv_data:
            if row['midi_file_path'] in files_to_update_paths:
                row['status'] = new_status
        
        # Write back to CSV
        with open(self.csv_tracker_path, 'w', newline='') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
    
    def _recreate_training_dataset(self):
        """Recreate training dataset with updated file list."""
        csv_data = self._load_csv_tracker()
        
        # Get training files
        training_files = [row for row in csv_data if row['status'] == 'training']
        
        # Create new training records
        training_records = []
        for row in training_files:
            training_records.append({
                'midi_file_path': row['midi_file_path'],
                'style_file_path': row['style_file_path']
            })
        
        # Update training pkl file
        import pickle
        training_pkl_path = self.data_module.cfg.data_dir
        training_data = {
            'metadata': {'purpose': 'progressive_training', 'total_files': len(training_records)},
            'data_records': training_records
        }
        
        with open(training_pkl_path, 'wb') as f:
            pickle.dump(training_data, f)
        
        overall_logger.info(f"Updated training dataset with {len(training_records)} files")


def create_csv_tracker(data_dir: str, csv_path: str, val_split: float = 0.1):
    """Create CSV tracker for dataset management."""
    import pickle
    
    # Load original data
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    
    all_records = data['data_records']
    random.shuffle(all_records)
    
    # Split into validation and remaining
    val_size = int(len(all_records) * val_split)
    val_records = all_records[:val_size]
    remaining_records = all_records[val_size:]
    
    # Create CSV entries
    csv_entries = []
    
    # Add validation files
    for record in val_records:
        csv_entries.append({
            'midi_file_path': record['midi_file_path'],
            'style_file_path': record['style_file_path'],
            'status': 'validation'
        })
    
    # Add remaining files (to be evaluated)
    for record in remaining_records:
        csv_entries.append({
            'midi_file_path': record['midi_file_path'],
            'style_file_path': record['style_file_path'],
            'status': 'remaining'
        })
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['midi_file_path', 'style_file_path', 'status'])
        writer.writeheader()
        writer.writerows(csv_entries)
    
    # Create initial training pkl (empty)
    training_pkl_path = data_dir.replace('.pkl', '_training.pkl')
    training_data = {
        'metadata': {'purpose': 'progressive_training', 'total_files': 0},
        'data_records': []
    }
    with open(training_pkl_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    # Create validation pkl
    val_pkl_path = data_dir.replace('.pkl', '_validation.pkl')
    val_data = {
        'metadata': {'purpose': 'validation', 'total_files': len(val_records)},
        'data_records': val_records
    }
    with open(val_pkl_path, 'wb') as f:
        pickle.dump(val_data, f)
    
    overall_logger.info(f"Created CSV tracker with {len(val_records)} validation files and {len(remaining_records)} remaining files")
    return training_pkl_path, val_pkl_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. LightningModule with Flash + Macaron
# ─────────────────────────────────────────────────────────────────────────────
class MidiClassifier(pl.LightningModule):
    def __init__(self,
        vocab_size: int,
        n_classes: int,
        lr: float,
        max_length: int,
        dropout: float,
        pad_id: int,
        hidden: int = 512,
        layers: int = 8,
        heads: int = 8,
        macaron: bool = True,
        attn_flash: bool = True,
        ff_glu: bool = False,
        layer_dropout: float = 0.1,
        attn_dropout: float = None,
        ff_dropout: float = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Use passed dropout values or fall back to main dropout
        attn_dropout = attn_dropout if attn_dropout is not None else dropout
        ff_dropout = ff_dropout if ff_dropout is not None else dropout

        # Build TransformerWrapper with FlashAttention + Macaron-style FFNs
        self.transformer = TransformerWrapper(
            num_tokens    = vocab_size,
            max_seq_len   = max_length,
            emb_dropout   = dropout,
            macaron       = macaron,                 # Enables two half‐step FFNs per layer
            attn_layers   = Encoder(
                dim             = hidden,
                depth           = layers,
                heads           = heads,
                ff_dropout      = ff_dropout,
                attn_dropout    = attn_dropout,
                layer_dropout   = layer_dropout,
                ff_glu          = ff_glu,
                use_scalenorm   = False,
                attn_flash      = attn_flash,        # FlashAttention kernel
            )
        )

        # Final token‐wise classifier
        self.to_logits = torch.nn.Linear(hidden, n_classes, bias=False)
        self.train_acc  = Accuracy(task='multiclass', num_classes=n_classes,
                                   ignore_index=IGNORE_LABEL_IDX)
        self.val_acc    = Accuracy(task='multiclass', num_classes=n_classes,
                                   ignore_index=IGNORE_LABEL_IDX)

    def forward(self, x, mask=None):
        hidden = self.transformer(x, mask=mask)
        return self.to_logits(hidden)

    def common_step(self, batch, stage):
        x, mask, labels = (batch['input_ids'],
                           batch['attention_mask'],
                           batch['form_label'])
        logits = self(x, mask)
        loss   = F.cross_entropy(logits.view(-1, self.hparams.n_classes),
                                 labels.view(-1),
                                 ignore_index=IGNORE_LABEL_IDX)
        acc = self.train_acc if stage=='train' else self.val_acc
        acc(logits.view(-1, self.hparams.n_classes), labels.view(-1))
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)
        self.log(f'{stage}_acc',  acc,   prog_bar=True, on_epoch=True)
        if stage=='train':
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'],
                     on_step=True, prog_bar=True)
        return loss

    def training_step(self, batch, idx):    return self.common_step(batch, 'train')
    def validation_step(self, batch, idx):  return self.common_step(batch, 'val')

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        steps  = self.trainer.estimated_stepping_batches
        warmup = int(0.1 * steps)
        sched  = get_cosine_schedule_with_warmup(opt, warmup, steps)
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'interval': 'step'}
        }


def main():
    parser = ArgumentParser()
    # Data args
    parser.add_argument('--data_dir', type=str, default='datasets/synth_midi_90k.pkl')
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--limit_data', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.1)
    # Model args
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    # Hyperparameters for MidiClassifier
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--macaron', action='store_true', help='Use macaron style transformer blocks')
    parser.add_argument('--attn_flash', action='store_true', help='Use flash attention')
    parser.add_argument('--ff_glu', action='store_true', help='Use GLU in feedforward')
    parser.add_argument('--layer_dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--ff_dropout', type=float, default=0.1)
    # Training args
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--accum_grads', type=int, default=4)
    parser.add_argument('--confidence_threshold', type=float, default=0.95)
    # Hardware args
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--nodes', type=int, default=1)
    # Logging args
    parser.add_argument('--wandb_project', type=str, default='midi_style')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default=None)

    args = parser.parse_args()

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    )

    # reproducibility
    set_seed(42)

    # Create CSV tracker and split datasets
    csv_tracker_path = args.data_dir.replace('.pkl', '_tracker.csv')
    
    if not os.path.exists(csv_tracker_path):
        overall_logger.info("Creating CSV tracker and splitting dataset...")
        training_pkl_path, val_pkl_path = create_csv_tracker(
            args.data_dir, csv_tracker_path, args.val_split
        )
    else:
        overall_logger.info("Using existing CSV tracker...")
        training_pkl_path = args.data_dir.replace('.pkl', '_training.pkl')
        val_pkl_path = args.data_dir.replace('.pkl', '_validation.pkl')

    # DataModule for training (initially empty or with previously added files)
    train_data_cfg = DataCfg(
        data_dir=training_pkl_path,
        mode='synthetic',
        task='classification',
        max_len=args.max_length,
        seq_limit=args.limit_data,
        shuffle_records=True,
        skip_long_after_tokenization=True,
        augment=AugmentCfg(enable=True, pitch=5, velocity=10, tempo=0.2, mixup=False),
    )
    
    # DataModule for validation
    val_data_cfg = DataCfg(
        data_dir=val_pkl_path,
        mode='synthetic',
        task='classification',
        max_len=args.max_length,
        shuffle_records=False,
        skip_long_after_tokenization=True,
        augment=AugmentCfg(enable=False),  # No augmentation for validation
    )

    dm = MidiDataModule(
        cfg=train_data_cfg,
        tokenizer=MusicTokenizerWithStyle(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_cfg=val_data_cfg,  # Pass validation config separately
    )

    # Model
    model = MidiClassifier(
        vocab_size=MusicTokenizerWithStyle().vocab_size,
        n_classes=args.n_classes,
        lr=args.lr,
        max_length=args.max_length,
        dropout=args.dropout_rate,
        pad_id=MusicTokenizerWithStyle().pad_id,
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        macaron=args.macaron,
        attn_flash=args.attn_flash,
        ff_glu=args.ff_glu,
        layer_dropout=args.layer_dropout,
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout,
    )

    # Logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        log_model=False,
    )
    
    # Create checkpoint directory based on wandb run name
    run_name = wandb_logger.name
    checkpoint_dir = os.path.join('checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=3, save_last=False,
        filename='epoch{epoch:02d}-val{val_loss:.2f}', dirpath=checkpoint_dir,
    )
    es_cb = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    gpu_cb = GPUMemoryCallback()
    progress_cb = DatasetProgressCallback(
        csv_tracker_path=csv_tracker_path,
        data_module=dm,
        confidence_threshold=args.confidence_threshold
    )

    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[ckpt_cb, es_cb, gpu_cb, progress_cb],
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        num_nodes=args.nodes,
        strategy='ddp_find_unused_parameters_true' if args.gpus > 1 else None,
        precision='bf16-mixed',
        accumulate_grad_batches=args.accum_grads,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    # Fit
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt)

    # Finish
    overall_logger.info('Training complete')


if __name__ == '__main__':
    main()