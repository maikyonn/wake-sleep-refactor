from __future__ import annotations
import os, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import math # Not strictly needed here anymore unless for complex splitting
import random
import argparse

from src.StariaTokenizer import (
    MusicTokenizerWithStyle,
    PROMPT_START_TOKEN, PROMPT_END_TOKEN,
    A_SECTION_TOKEN, B_SECTION_TOKEN, C_SECTION_TOKEN, D_SECTION_TOKEN,
    FORM_LABEL_MAP, IGNORE_LABEL_IDX, SPECIAL_TOKENS
)
from src.utils_new import music_style_from_labels # Assuming this utility exists

logger = logging.getLogger(__name__)

# --- Configs --- (Copied from your previous DataModule)
@dataclass
class AugmentCfg:
    enable: bool = True
    pitch: int = 0
    velocity: int = 0
    tempo: float = 0.0
    mixup: bool = False

@dataclass
class DataCfg:
    data_dir: str  # Path to the .pkl file containing list of path dictionaries (for training)
    val_data_dir: Optional[str] = None # Path to .pkl file for validation
    mode: str = "synthetic"
    task: str = "generative"
    use_snippet: bool = False
    max_len: int = 4096 # Max token sequence length AFTER tokenization
    seq_limit: Optional[int] = None # Max number of samples to use from the .pkl file
    shuffle_records: bool = True # Shuffle the list of records (paths) during setup
    skip_long_after_tokenization: bool = True # Skip if tokenized sequence > max_len
    augment: Optional[AugmentCfg] = None

# --- Helpers ---
_SECTION_TOK = {"A": A_SECTION_TOKEN, "B": B_SECTION_TOKEN, "C": C_SECTION_TOKEN, "D": D_SECTION_TOKEN}
_SNIPPET_LEN = 256

def get_style_labels_from_file(label_path: str) -> Optional[List[str]]:
    """Reads a style file and returns the style characters (copied from your preprocess)."""
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()
        if not content:
            # logger.warning(f"Empty style file: {label_path}") # Can be too verbose
            return None
        return list(content)
    except Exception as e:
        logger.error(f"Error processing style file {label_path}: {e}", exc_info=False)
        return None

# --- On-Demand Map-Style Dataset ---
class OnDemandMidiDataset(Dataset):
    def __init__(self, cfg: DataCfg, tokenizer: MusicTokenizerWithStyle, path_pkl_file: str, is_validation_set: bool = False):
        self.cfg = cfg
        self.tok = tokenizer
        self.path_pkl_file = path_pkl_file
        self.is_validation_set = is_validation_set # To control augmentation if needed differently for val
        self.pad_id = tokenizer.pad_id

        logger.info(f"Initializing OnDemandMidiDataset with path_pkl_file: {path_pkl_file}")
        try:
            with open(path_pkl_file, "rb") as f:
                loaded_data = pickle.load(f)
            # Expecting new format: {"metadata": ..., "data_records": List[Dict]}
            self._records = loaded_data["data_records"] 
            self.metadata = loaded_data.get("metadata", {})
            logger.info(f"  Loaded {len(self._records)} path records. Metadata: {self.metadata}")
        except Exception as e:
            logger.error(f"Failed to load or parse path PKL file {path_pkl_file}: {e}", exc_info=True)
            self._records = []
            self.metadata = {}
        
        if self.cfg.seq_limit is not None and self.cfg.seq_limit < len(self._records):
            logger.info(f"  Using seq_limit: {self.cfg.seq_limit} samples out of {len(self._records)}.")
            # If shuffling records is enabled in cfg, it should happen before this limit is applied.
            # This is handled in MidiDataModule.setup()
            self._records = self._records[:self.cfg.seq_limit]


        # Augmentation setup
        current_augment_cfg = cfg.augment
        if current_augment_cfg is None:
            if cfg.task == "contrastive":
                current_augment_cfg = AugmentCfg(enable=True, pitch=5, velocity=10, tempo=0, mixup=False)
            elif cfg.task == "generative":
                # Enable augmentation only for training set by default for generative
                current_augment_cfg = AugmentCfg(enable=False, pitch=5, velocity=10, tempo=0.1, mixup=False)
            else:
                current_augment_cfg = AugmentCfg(enable=False)
        self.augment_cfg = current_augment_cfg
        
        self.aug_fns = []
        if self.augment_cfg.enable:
            logger.info(f"  Augmentation ENABLED for this dataset instance: {self.augment_cfg}")
            base = tokenizer._tokenizer
            if self.augment_cfg.pitch: self.aug_fns.append(base.export_pitch_aug(self.augment_cfg.pitch))
            if self.augment_cfg.velocity: self.aug_fns.append(base.export_velocity_aug(self.augment_cfg.velocity))
            if self.augment_cfg.tempo: self.aug_fns.append(base.export_tempo_aug(self.augment_cfg.tempo, self.augment_cfg.mixup))
        else:
            logger.info(f"  Augmentation DISABLED for this dataset instance.")


    def _apply_aug(self, tokens_str_list: List[str]) -> List[str]:
        if not self.aug_fns: return tokens_str_list
        # (Copied from previous FlexibleMidiDataset - ensure SPECIAL_TOKENS is accessible)
        special_indices = [i for i, token_str in enumerate(tokens_str_list) if token_str in SPECIAL_TOKENS]
        special_tokens_found = [tokens_str_list[i] for i in special_indices]
        regular_tokens = [token_str for i, token_str in enumerate(tokens_str_list) if i not in special_indices]
        
        augmented_regular_tokens = regular_tokens
        for fn in self.aug_fns:
            augmented_regular_tokens = fn(augmented_regular_tokens)
        
        result_tokens = list(augmented_regular_tokens)
        for i, (special_idx, special_token_str) in enumerate(zip(special_indices, special_tokens_found)):
            if special_idx <= len(result_tokens): result_tokens.insert(special_idx, special_token_str)
            else: result_tokens.append(special_token_str)
        return result_tokens

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx >= len(self._records):
            raise IndexError("Index out of bounds")
        
        record_paths = self._records[idx]
        midi_file_path = record_paths.get("midi_file_path")
        style_file_path = record_paths.get("style_file_path") # Will be None for "real" mode

        if not midi_file_path or not os.path.exists(midi_file_path):
            logger.warning(f"MIDI file path not found or does not exist: {midi_file_path} for record index {idx}. Skipping.")
            return None # Skip this sample

        # 1. Load from file paths and initial tokenization (string tokens)
        tokens_str_list = self.tok.tokenize_from_file(midi_file_path)
        if tokens_str_list is None or not tokens_str_list:
            # logger.debug(f"Tokenization failed or yielded empty for {midi_file_path}. Skipping.")
            return None

        style_labels_str_list = None
        if self.cfg.mode == "synthetic":
            if not style_file_path or not os.path.exists(style_file_path):
                logger.warning(f"Style file path not found or does not exist: {style_file_path} for synthetic record {midi_file_path}. Skipping.")
                return None
            style_labels_str_list = get_style_labels_from_file(style_file_path)
            if style_labels_str_list is None:
                # logger.debug(f"Failed to get style labels for {style_file_path}. Skipping.")
                return None
            if len(tokens_str_list) != len(style_labels_str_list):
                # logger.warning(f"Length mismatch: MIDI tokens {len(tokens_str_list)}, Style labels {len(style_labels_str_list)} for {midi_file_path}. Skipping.")
                return None
        
        # 2. Augmentation (on string tokens)
        if self.augment_cfg.enable:
            tokens_str_list = self._apply_aug(tokens_str_list)
            # Note: style_labels are typically not augmented, or require careful handling if they are.

        # 3. Task-specific processing and numerical encoding
        # This logic is similar to _process_record from FlexibleMidiDataset, adapted for on-demand loading.

        # -- Generative Full Task --
        if self.cfg.task == "generative" and not (self.cfg.mode == "synthetic" and self.cfg.use_snippet):
            ids_list = self.tok.encode(tokens_str_list)
            if len(ids_list) > self.cfg.max_len:
                if self.cfg.skip_long_after_tokenization: return None
                ids_list = ids_list[:self.cfg.max_len]
            if not ids_list: return None
            return {"input_ids": torch.tensor(ids_list, dtype=torch.long),
                    "attention_mask": torch.ones(len(ids_list), dtype=torch.long)}

        # -- Generative Synthetic with Snippet --
        elif self.cfg.task == "generative" and self.cfg.mode == "synthetic" and self.cfg.use_snippet:
            if style_labels_str_list is None: return None # Should have been caught earlier but defensive check

            # For snippets, encoder uses augmented tokens, decoder uses original (non-augmented)
            # `tokens_str_list` is already augmented here if aug is enabled.
            # We need original tokens for decoder target. Let's re-tokenize original MIDI for decoder if augmentation happened.
            original_midi_tokens_for_decoder = self.tok.tokenize_from_file(midi_file_path)
            if original_midi_tokens_for_decoder is None: return None
            
            tokens_for_snippet_aug = self.tok.remove_instrument_prefix(tokens_str_list) # Augmented for encoder prompt

            music_style = music_style_from_labels(style_labels_str_list)
            enc_prompt_tokens = [PROMPT_START_TOKEN]
            runs = []
            start_idx = 0
            for i in range(1, len(style_labels_str_list)):
                if style_labels_str_list[i] != style_labels_str_list[i-1]:
                    runs.append((style_labels_str_list[i-1], start_idx, i-1))
                    start_idx = i
            if style_labels_str_list: runs.append((style_labels_str_list[-1], start_idx, len(style_labels_str_list)-1))
            else: return None

            for style_char in music_style:
                for lab_r, s_r, e_r in runs:
                    if lab_r == style_char:
                        enc_prompt_tokens.append(_SECTION_TOK[lab_r])
                        seg_end = min(e_r + 1, s_r + _SNIPPET_LEN)
                        if s_r < len(tokens_for_snippet_aug):
                            seg = tokens_for_snippet_aug[s_r : min(seg_end, len(tokens_for_snippet_aug))]
                            enc_prompt_tokens.extend(seg)
                        break
            enc_prompt_tokens.append(PROMPT_END_TOKEN)
            
            enc_ids_list = self.tok.encode(enc_prompt_tokens)
            dec_target_tokens_no_prefix = self.tok.remove_instrument_prefix(original_midi_tokens_for_decoder)
            dec_ids_list = self.tok.encode(dec_target_tokens_no_prefix)

            for id_list_ref_wrapper in [[enc_ids_list], [dec_ids_list]]:
                current_list = id_list_ref_wrapper[0]
                if len(current_list) > self.cfg.max_len:
                    if self.cfg.skip_long_after_tokenization: return None
                    id_list_ref_wrapper[0] = current_list[:self.cfg.max_len]
                if not id_list_ref_wrapper[0]: return None
            enc_ids_list, dec_ids_list = enc_ids_list, dec_ids_list # Reassign

            return {
                "encoder_ids": torch.tensor(enc_ids_list, dtype=torch.long),
                "encoder_mask": torch.ones(len(enc_ids_list), dtype=torch.long),
                "decoder_ids": torch.tensor(dec_ids_list, dtype=torch.long),
                "decoder_mask": torch.ones(len(dec_ids_list), dtype=torch.long)
            }

        # -- Contrastive Task --
        elif self.cfg.task == "contrastive":
            if style_labels_str_list is None: return None

            # For contrastive, create two views from the original, then augment each.
            original_midi_tokens = self.tok.tokenize_from_file(midi_file_path) # Fresh load of original
            if original_midi_tokens is None: return None
            tokens_no_prefix = self.tok.remove_instrument_prefix(original_midi_tokens)
            
            tokens_view1 = self._apply_aug(list(tokens_no_prefix)) # Augment view 1
            tokens_view2 = self._apply_aug(list(tokens_no_prefix)) # Augment view 2 (could be different if aug is stochastic)

            runs = [] # Calculated from original style labels
            start_idx = 0
            for i in range(1, len(style_labels_str_list)):
                if style_labels_str_list[i] != style_labels_str_list[i-1]:
                    runs.append((style_labels_str_list[i-1], start_idx, i-1))
                    start_idx = i
            if style_labels_str_list: runs.append((style_labels_str_list[-1], start_idx, len(style_labels_str_list)-1))
            else: return None
            if not runs: return None
            
            _lab, s, e = runs[0]
            snippet_view1 = tokens_view1[s:min(e + 1, s + _SNIPPET_LEN)]
            snippet_view2 = tokens_view2[s:min(e + 1, s + _SNIPPET_LEN)]
            e1 = self.tok.encode(snippet_view1)
            e2 = self.tok.encode(snippet_view2)

            music_style = music_style_from_labels(style_labels_str_list)
            prompt_base = [PROMPT_START_TOKEN] # Build from original, non-augmented tokens
            for style_char in music_style:
                for lab_r, s_r, e_r in runs:
                    if lab_r == style_char:
                        prompt_base.append(_SECTION_TOK[lab_r])
                        seg = tokens_no_prefix[s_r: min(e_r + 1, s_r + _SNIPPET_LEN)]
                        prompt_base.extend(seg)
                        break
            prompt_base.append(PROMPT_END_TOKEN)
            prompt_view1 = self._apply_aug(list(prompt_base))
            prompt_view2 = self._apply_aug(list(prompt_base))
            p1 = self.tok.encode(prompt_view1)
            p2 = self.tok.encode(prompt_view2)

            lists_to_check = [[e1], [e2], [p1], [p2]]
            for wrapped_list in lists_to_check:
                current_list_val = wrapped_list[0]
                if len(current_list_val) > self.cfg.max_len:
                    if self.cfg.skip_long_after_tokenization: return None
                    wrapped_list[0] = current_list_val[:self.cfg.max_len]
                if not wrapped_list[0]: return None
            e1,e2,p1,p2 = lists_to_check[0][0], lists_to_check[1][0], lists_to_check[2][0], lists_to_check[3][0]

            return dict(
                x1_local=torch.tensor(e1), x2_local=torch.tensor(e2), mask_local=torch.ones(len(e1),dtype=torch.long),
                x1_prompt=torch.tensor(p1), x2_prompt=torch.tensor(p2), mask_prompt=torch.ones(len(p1),dtype=torch.long)
            )

        # -- Classification Task --
        elif self.cfg.task == "classification":
            # `tokens_str_list` is already (maybe) augmented
            form = Path(midi_file_path).stem.split("_")[0] # Use midi_file_path for form label derivation
            lbl = FORM_LABEL_MAP.get(form)
            if lbl is None: return None
            
            ids_list = self.tok.encode(tokens_str_list)
            if len(ids_list) > self.cfg.max_len:
                if self.cfg.skip_long_after_tokenization: return None
                ids_list = ids_list[:self.cfg.max_len]
            if not ids_list: return None
            return {"input_ids": torch.tensor(ids_list, dtype=torch.long),
                    "attention_mask": torch.ones(len(ids_list), dtype=torch.long),
                    "form_label": torch.tensor(lbl, dtype=torch.long)}
        
        logger.warning(f"Unhandled task/mode in OnDemandMidiDataset for {midi_file_path}: {self.cfg.task}/{self.cfg.mode}")
        return None

# --- Collate Function (can be largely the same as before) ---
def midi_collate_mapstyle(batch: List[Optional[Dict[str, Any]]], tokenizer: MusicTokenizerWithStyle) -> Optional[Dict[str, Any]]:
    batch = [b for b in batch if b is not None and isinstance(b, dict) and b] 
    if not batch:
        # logger.warning("Collate: received an empty batch after filtering. Returning None.")
        raise StopIteration("Collate: received an empty batch after filtering. Returning None.")

    first_item_keys = batch[0].keys()
    result = {}

    def safe_pad_sequence(key: str, pad_val: Union[int, float], default_dtype=torch.long) -> torch.Tensor:
        tensor_list = [b[key] for b in batch if key in b and isinstance(b[key], torch.Tensor) and b[key].numel() > 0]
        if not tensor_list:
            # This case means for this specific key, all items in the batch were empty or invalid.
            # The entire batch might still be valid if other keys are fine.
            # Returning a (0,0) or (0,X) tensor for this key.
            # If this key is essential, the model might fail. This indicates a data problem.
            # Let's try (BatchSize, 0) to match pad_sequence behavior for empty items within a batch for this key
            return torch.empty((len(batch), 0), dtype=default_dtype)
        return pad_sequence(tensor_list, batch_first=True, padding_value=pad_val)

    if "x1_local" in first_item_keys: # Contrastive
        for key in ['x1_local', 'x2_local', 'x1_prompt', 'x2_prompt']: result[key] = safe_pad_sequence(key, tokenizer.pad_id)
        for key in ['mask_local', 'mask_prompt']: result[key] = safe_pad_sequence(key, 0)
        if result['x1_local'].shape[1] == 0 and result['x1_local'].shape[0] > 0 :  # All sequences for x1_local were empty
            logger.warning("Contrastive batch: all x1_local sequences became empty after processing. Batch might be invalid.")
            # Potentially return None or a more specific empty structure if this is fatal
        return result
    elif 'form_label' in first_item_keys: # Classification
        result['input_ids'] = safe_pad_sequence('input_ids', tokenizer.pad_id)
        result['attention_mask'] = safe_pad_sequence('attention_mask', 0)
        labels_list = [b['form_label'] for b in batch if 'form_label' in b] # Labels usually don't need padding if scalar
        result['form_label'] = torch.stack(labels_list) if labels_list else torch.tensor([], dtype=torch.long)
        # Check if input_ids are all empty after padding
        if result['input_ids'].shape[1] == 0 and result['input_ids'].shape[0] > 0:
            logger.warning("Classification batch: all input_ids sequences became empty. Batch might be invalid.")
        return result
    elif 'encoder_ids' in first_item_keys: # Snippet Generative
        for key in ['encoder_ids', 'decoder_ids']: result[key] = safe_pad_sequence(key, tokenizer.pad_id)
        for key in ['encoder_mask', 'decoder_mask']: result[key] = safe_pad_sequence(key, 0)
        if result['encoder_ids'].shape[1] == 0 and result['encoder_ids'].shape[0] > 0 :
            logger.warning("Snippet Gen batch: all encoder_ids sequences became empty. Batch might be invalid.")
        return result
    elif 'input_ids' in first_item_keys: # Full Generative
        result['input_ids'] = safe_pad_sequence('input_ids', tokenizer.pad_id)
        result['attention_mask'] = safe_pad_sequence('attention_mask', 0)
        if result['input_ids'].shape[1] == 0 and result['input_ids'].shape[0] > 0 :
            logger.warning("Generative batch: all input_ids sequences became empty. Batch might be invalid.")
        return result
    
    logger.error(f"Collate: Could not determine batch structure. Keys: {first_item_keys if batch else 'Empty Batch'}. Returning None.")
    return None

# --- CLI Quick-check ---
if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s %(name)s %(process)d: %(message)s')
    parser = argparse.ArgumentParser(description="Test OnDemandMidiDataset and DataModule")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to .pkl file of training file paths")
    parser.add_argument("--val_data_dir", type=str, default=None, help="Path to .pkl file of validation file paths")
    parser.add_argument("--task", default="generative")
    parser.add_argument("--mode", default="synthetic")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seq_limit", type=int, default=None, help="Limit total samples from pkl to use") # For map-style, this limits dataset size
    parser.add_argument("--use_snippet", action="store_true")

    cli_args = parser.parse_args()
    tok = MusicTokenizerWithStyle()
    
    data_cfg_main = DataCfg(
        data_dir=cli_args.data_dir, val_data_dir=cli_args.val_data_dir,
        mode=cli_args.mode, task=cli_args.task, max_len=cli_args.max_len,
        seq_limit=cli_args.seq_limit, 
        shuffle_records=True, # Test with shuffling records
        skip_long_after_tokenization=False, # See truncated items
        use_snippet=cli_args.use_snippet
    )
    dm_main = MidiDataModule(cfg=data_cfg_main, tokenizer=tok, batch_size=cli_args.batch_size, num_workers=cli_args.num_workers)
    dm_main.setup(stage='fit')
    
    for split_name_main in ['train', 'val']:
        logger.info(f"\n--- Checking {split_name_main} dataloader ---")
        loader_main = getattr(dm_main, f"{split_name_main}_dataloader")()
        if loader_main:
            logger.info(f"DataLoader for {split_name_main} created. Length: {len(loader_main) if hasattr(loader_main, '__len__') and loader_main.dataset else 'Unknown/Iterable'}")
            item_count_to_check = 2 
            try:
                for i_main, batch_main in enumerate(loader_main):
                    if batch_main is None:
                        logger.warning(f"{split_name_main.capitalize()} Batch {i_main+1} is None (skipped by collate).")
                        continue
                    logger.info(f"{split_name_main.capitalize()} Batch {i_main+1} keys: {batch_main.keys()}")
                    for k_main, v_main in batch_main.items():
                        if isinstance(v_main, torch.Tensor): 
                            logger.info(f"  {k_main}: shape={v_main.shape}, dtype={v_main.dtype}, device={v_main.device}")
                    if i_main >= item_count_to_check -1 : break 
                logger.info(f"{split_name_main.capitalize()} dataloader iteration test successful for {item_count_to_check} batches.")
            except Exception as e_main:
                logger.error(f"Error iterating {split_name_main}_dataloader: {e_main}", exc_info=True)
        else:
            logger.info(f"{split_name_main.capitalize()} dataloader is None.")
    logger.info("\nMidiDataModule CLI check completed!")