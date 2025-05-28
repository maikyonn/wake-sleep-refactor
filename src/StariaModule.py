r"""StariaModule.py
=======================
⚡ **PyTorch‑Lightning wrapper** for the Staria model.

The StariaModel combines an encoder and decoder for sequence-to-sequence tasks.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F # Keep for potential future use, though _compute_loss uses nn.CrossEntropyLoss
import pytorch_lightning as pl
from src.StariaTokenizer import MusicTokenizerWithStyle # Assuming this is your tokenizer class
from x_transformers import TransformerWrapper, Decoder, Encoder

def _shift_right(x: torch.Tensor, pad_id: int, bos_id: int) -> torch.Tensor:
    """
    Shift tensor right by one position for decoder training.
    Inserts BOS at the beginning and removes the last token.
    """
    shifted = torch.full_like(x, pad_id)
    shifted[:, 0] = bos_id
    shifted[:, 1:] = x[:, :-1].clone() # x[:, :-1] ensures the output length matches original if x was full
    return shifted

class StariaModel(pl.LightningModule):
    def __init__(
        self,
        tokenizer: MusicTokenizerWithStyle, # Pass the tokenizer instance
        lr: float = 3e-4, # Adjusted default learning rate
        encoder_dim: int = 1536,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        decoder_dim: int = 1536,
        decoder_depth: int = 16,
        decoder_heads: int = 24,
        max_len: int = 4096,
        # Add task as a parameter if model behavior needs to change fundamentally based on it
        # current_task: str = "generative_snippet", # Example
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.lr = lr
        # self.current_task = current_task # Store if needed

        # Get special token IDs from tokenizer
        self.pad_id = getattr(tokenizer, 'pad_id', 0) # Make sure these attributes exist on your tokenizer
        self.bos_id = getattr(tokenizer, 'bos_id', 1)
        self.eos_id = getattr(tokenizer, 'eos_id', 2)
        self.vocab_size = getattr(tokenizer, 'vocab_size')

        if self.vocab_size is None:
            raise ValueError("Tokenizer must have a 'vocab_size' attribute.")

        # Build encoder
        self.encoder = TransformerWrapper(
            num_tokens=self.vocab_size,
            max_seq_len=max_len,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads,
                attn_flash=True, # Enable Flash Attention if available and desired
            )
        )
        
        # Build decoder with cross-attention
        self.decoder = TransformerWrapper(
            num_tokens=self.vocab_size,
            max_seq_len=max_len,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=True,
                attn_flash=True, # Enable Flash Attention
            )
        )
        
        # save_hyperparameters will save all args passed to __init__
        # Be cautious if tokenizer is a very complex object; often it's better to log its config.
        self.save_hyperparameters(ignore=['tokenizer']) # Ignore tokenizer object itself

    def configure_optimizers(self):
        # Consider adding weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        # Example scheduler: linear warmup and decay (needs total steps)
        # scheduler = get_linear_schedule_with_warmup(...)
        return optimizer

    def forward(self, src_ids: torch.Tensor, 
                decoder_input_ids: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                decoder_input_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for training/evaluation where decoder inputs are provided.
        Args:
            src_ids: Source sequence for the encoder.
            decoder_input_ids: Target sequence for the decoder (e.g., shifted right).
            src_mask: Mask for src_ids.
            decoder_input_mask: Mask for decoder_input_ids.
        """
        if src_mask is not None and not src_mask.dtype == torch.bool:
            src_mask = src_mask.bool()
        if decoder_input_mask is not None and not decoder_input_mask.dtype == torch.bool:
            decoder_input_mask = decoder_input_mask.bool()

        # Encoder processes the source sequence
        # return_embeddings=True gives the sequence of hidden states
        encoder_context = self.encoder(src_ids, mask=src_mask, return_embeddings=True)
        
        # Decoder uses the encoder's output as context and processes the decoder_input_ids
        # The decoder's internal causal mask handles autoregressive property for decoder_input_ids.
        # The decoder_input_mask here handles padding in the decoder_input_ids.
        # The context_mask (src_mask) handles padding in the encoder_context for cross-attention.
        output_logits = self.decoder(
            decoder_input_ids, 
            mask=decoder_input_mask, 
            context=encoder_context
        )
        return output_logits

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        """Compute cross entropy loss, ignoring padding tokens in labels."""
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        # logits: (batch_size, seq_len, vocab_size)
        # labels: (batch_size, seq_len)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss
    
    def _extract_src_tgt_from_batch(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts source, source_mask, target_labels, and target_labels_mask from batch.
        target_labels are the actual ground truth tokens the decoder should predict.
        target_labels_mask corresponds to target_labels.
        """
        src_ids, src_mask, target_labels, target_labels_mask = None, None, None, None

        if "encoder_ids" in batch and "decoder_ids" in batch:
            # Case 1: Explicit encoder_ids and decoder_ids (e.g., generative snippet task)
            src_ids = batch["encoder_ids"]
            src_mask = batch.get("encoder_mask")
            target_labels = batch["decoder_ids"]
            target_labels_mask = batch.get("decoder_mask")
            # logger.debug("Using encoder_ids and decoder_ids from batch.")

        elif "input_ids" in batch:
            # Case 2: Only input_ids provided (e.g., generative full task)
            # Encoder sees input_ids, Decoder's target is also input_ids.
            src_ids = batch["input_ids"]
            src_mask = batch.get("attention_mask")
            target_labels = batch["input_ids"] # Target for the decoder to generate
            target_labels_mask = batch.get("attention_mask") # Mask for the target_labels
            # logger.debug("Using input_ids for both encoder and decoder targets.")
            
        # Add more cases here if other tasks like "classification" or "contrastive"
        # are meant to be trained with this seq2seq loss structure.
        # For classification, target_labels would be batch["form_label"], and loss calculation would change.
        # For contrastive, the entire step logic would be different.

        else:
            raise ValueError(
                "Batch must contain ('encoder_ids' and 'decoder_ids') OR 'input_ids'. "
                f"Received keys: {list(batch.keys())}"
            )
        
        # Ensure masks are present or create default ones (all ones, meaning no masking)
        if src_mask is None:
            src_mask = torch.ones_like(src_ids, device=src_ids.device, dtype=torch.long)
        if target_labels_mask is None:
            target_labels_mask = torch.ones_like(target_labels, device=target_labels.device, dtype=torch.long)
            
        return src_ids, src_mask, target_labels, target_labels_mask

    def _common_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        src_ids, src_mask, target_labels, _ = self._extract_src_tgt_from_batch(batch)
        # target_labels_mask from _extract_src_tgt_from_batch is not directly used here,
        # as decoder_input_mask is derived from shifted targets.
        # However, it's good to have it if other loss components needed it.

        # 1. Prepare decoder_input_ids (teacher forcing: shifted right version of target_labels)
        decoder_input_ids = _shift_right(target_labels, pad_id=self.pad_id, bos_id=self.bos_id)
        
        # 2. Prepare decoder_input_mask (masks padding in decoder_input_ids)
        decoder_input_mask = decoder_input_ids.ne(self.pad_id) # True for non-pad tokens

        # Ensure masks are boolean for the model's forward pass
        # (TransformerWrapper might handle it, but explicit is safer)
        src_mask_bool = src_mask.bool()
        decoder_input_mask_bool = decoder_input_mask.bool()
        
        # 3. Forward pass
        logits = self(
            src_ids=src_ids, 
            decoder_input_ids=decoder_input_ids, 
            src_mask=src_mask_bool, 
            decoder_input_mask=decoder_input_mask_bool
        )

        # 4. Compute loss against the original (unshifted) target_labels
        loss = self._compute_loss(logits, target_labels)
        return loss
    
    # ---------------- training_step ----------------
    def training_step(self, batch, batch_idx):
        # ❷ This `if` should normally never trigger after the collate fix,
        #   but keep it as a safety net that returns a *real* tensor.
        if batch is None:
            print("training_step received None batch at batch_idx {batch_idx}. Skipping.")
            return torch.tensor(0., device=self.device, requires_grad=True)

        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        if batch is None:
            # logger.warning(f"validation_step received None batch at batch_idx {batch_idx}. Skipping.")
            return None
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss # Or return {"val_loss": loss} if you need to log other things

    def generate(self, prompt_ids: Optional[torch.Tensor] = None, 
                 max_new_tokens: int = 256, # Changed from max_length to max_new_tokens
                 temperature: float = 0.7,  # Common default
                 top_k: int = 0, 
                 top_p: float = 0.9, 
                 context_ids: Optional[torch.Tensor] = None, 
                 context_mask: Optional[torch.Tensor] = None, 
                 output_path: Optional[str] = None, 
                 debug: bool = False):
        self.eval() # Ensure model is in eval mode
        device = next(self.parameters()).device
        
        encoder_context = None
        if context_ids is not None:
            context_ids = context_ids.to(device)
            if context_mask is not None:
                context_mask = context_mask.to(device).bool()
            encoder_context = self.encoder(context_ids, mask=context_mask, return_embeddings=True)
        
        if prompt_ids is None:
            # Start with BOS token if no prompt is given
            prompt_ids = torch.tensor([[self.bos_id]], device=device, dtype=torch.long)
        else:
            prompt_ids = prompt_ids.to(device)
            
        batch_size = prompt_ids.shape[0]
        current_generated_ids = prompt_ids.clone() # This will hold the full sequence being generated
        
        generated_sequences = [[] for _ in range(batch_size)]

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Decoder input is the current state of generated sequence
                decoder_input_ids = current_generated_ids
                # Create decoder mask for padding (if any, though usually not needed in incremental generation)
                decoder_mask = decoder_input_ids.ne(self.pad_id).bool()

                logits = self.decoder(
                    decoder_input_ids, 
                    mask=decoder_mask,
                    context=encoder_context, 
                    context_mask=context_mask # Mask for cross-attention
                )
                
                next_token_logits = logits[:, -1, :].float() # Logits for the very last token predicted
                
                if temperature > 0 and temperature != 1.0: # Avoid division by zero or one
                    next_token_logits = next_token_logits / temperature
                
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0 # Always keep at least one token
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)
                
                if debug:
                    for b_idx in range(batch_size):
                        top_probs_debug, top_indices_debug = torch.topk(probs[b_idx], 5)
                        # logger.debug(f"Step {step}, Batch {b_idx} - Top 5 preds:")
                        # for i_debug, (idx_debug, prob_debug) in enumerate(zip(top_indices_debug.tolist(), top_probs_debug.tolist())):
                        #     token_str_debug = self.tokenizer.decode([idx_debug])
                        #     logger.debug(f"  {i_debug+1}. '{token_str_debug}' (id: {idx_debug}) - prob: {prob_debug:.4f}")
                        # logger.debug(f"  Selected: '{self.tokenizer.decode([next_token_ids[b_idx].item()])}' (id: {next_token_ids[b_idx].item()})\n")
                        pass # Placeholder for actual logging if needed

                current_generated_ids = torch.cat([current_generated_ids, next_token_ids], dim=-1)
                
                # Store the newly generated token for each item in the batch
                for b_idx in range(batch_size):
                    generated_sequences[b_idx].append(next_token_ids[b_idx].item())

                # Check if all sequences in batch ended with EOS
                if (next_token_ids == self.eos_id).all():
                    break
        
        # `current_generated_ids` contains the prompt + new tokens
        # `generated_sequences` contains only the new tokens (list of lists of ints)

        if output_path is not None and batch_size == 1:
            # Save the full sequence (prompt + generated)
            full_output_ids = current_generated_ids[0].cpu().tolist()
            try:
                self.tokenizer.ids_to_file(full_output_ids, output_path)
                # logger.info(f"Generated output saved to {output_path}")
            except Exception as e:
                # logger.error(f"Failed to save generated MIDI to {output_path}: {e}")
                pass # Placeholder

        return current_generated_ids # Return the full sequences including prompts


if __name__ == "__main__":
    # This __main__ block assumes you have MidiDataModule_mapstyle_ondemand.py in src
    # and the path PKL files generated by preprocess_makepaths_pkl.py
    from src.MidiDataModule_mapstyle_ondemand import DataCfg, MidiDataModule 
    
    print("StariaModel __main__ test block starting...")
    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    logger_test = logging.getLogger(__name__ + "_test")

    # --- Create dummy data for testing ---
    dummy_tokenizer = MusicTokenizerWithStyle() # Assuming this can be initialized simply
    logger_test.info(f"Dummy Tokenizer: vocab_size={dummy_tokenizer.vocab_size}, pad_id={dummy_tokenizer.pad_id}, bos_id={dummy_tokenizer.bos_id}")

    # Create dummy PKL files for paths
    dummy_train_path_pkl = "dummy_train_paths.pkl"
    dummy_val_path_pkl = "dummy_val_paths.pkl"
    
    # Create some dummy MIDI and style files if they don't exist (minimal content)
    os.makedirs("./dummy_data/midi", exist_ok=True)
    os.makedirs("./dummy_data/style", exist_ok=True)
    
    dummy_midi_content = "PIECE_START TIME_SIGNATURE=4/4 GENRE=DummyPop TRACK_START INST=0 DENSITY=1 NOTE_ON=60 VEL=80 NOTE_OFF=60 BAR NOTE_ON=64 VEL=80 NOTE_OFF=64 BAR TRACK_END PIECE_END" # Example REMI string
    dummy_style_content = "AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD" # Example style string

    train_path_records = []
    for i in range(5): # 5 dummy training files
        midi_f = f"./dummy_data/midi/train_song_{i}.mid" # Actual content doesn't matter as much as path existence for this test
        style_f = f"./dummy_data/style/train_song_{i}.txt"
        with open(midi_f, "w") as f: f.write(f"dummy midi content {i}") # Create empty files
        with open(style_f, "w") as f: f.write(dummy_style_content[:10+i]) # Vary length slightly
        train_path_records.append({"midi_file_path": os.path.abspath(midi_f), "style_file_path": os.path.abspath(style_f)})

    val_path_records = []
    for i in range(2): # 2 dummy validation files
        midi_f = f"./dummy_data/midi/val_song_{i}.mid"
        style_f = f"./dummy_data/style/val_song_{i}.txt"
        with open(midi_f, "w") as f: f.write(f"dummy val midi content {i}")
        with open(style_f, "w") as f: f.write(dummy_style_content[:12+i])
        val_path_records.append({"midi_file_path": os.path.abspath(midi_f), "style_file_path": os.path.abspath(style_f)})

    with open(dummy_train_path_pkl, "wb") as f: pickle.dump({"metadata": {}, "data_records": train_path_records}, f)
    with open(dummy_val_path_pkl, "wb") as f: pickle.dump({"metadata": {}, "data_records": val_path_records}, f)
    logger_test.info(f"Created dummy path PKL files: {dummy_train_path_pkl}, {dummy_val_path_pkl}")

    # --- Test with a specific task, e.g., "generative snippet" which uses encoder/decoder ---
    data_cfg_test = DataCfg(
        data_dir=dummy_train_path_pkl,
        val_data_dir=dummy_val_path_pkl,
        mode="synthetic", 
        task="generative", # This will determine keys from OnDemandMidiDataset
        use_snippet=True,  # Set to True to get encoder_ids and decoder_ids for this test
        max_len=128,      # Shorter max_len for quick testing
        seq_limit=None,   # Use all dummy samples
        shuffle_records=False, # No need to shuffle for this small test
        skip_long_after_tokenization=True,
    )
    
    dm_test = MidiDataModule(
        cfg=data_cfg_test,
        tokenizer=dummy_tokenizer,
        batch_size=1, # Test with batch size 1
        num_workers=0, # Test with 0 workers for simplicity
    )
    dm_test.setup(stage="fit")
    
    model_test = StariaModel(
        tokenizer=dummy_tokenizer,
        lr=1e-4,
        encoder_dim=64, encoder_depth=1, encoder_heads=2, # Tiny model for testing
        decoder_dim=64, decoder_depth=1, decoder_heads=2,
        max_len=data_cfg_test.max_len 
    )
    model_test.to('cpu') # Test on CPU if GPU not essential for this unit test
    logger_test.info("StariaModel and DataModule initialized for testing.")

    # Get a single sample batch from the dataloader
    train_dataloader = dm_test.train_dataloader()
    if not train_dataloader:
        logger_test.error("Failed to create train_dataloader for test.")
        exit()
        
    try:
        batch_test = next(iter(train_dataloader))
        if batch_test is None:
             logger_test.error("Test batch is None. Check data processing and collation.")
             exit()
    except Exception as e:
        logger_test.error(f"Error getting batch from test dataloader: {e}", exc_info=True)
        exit()

    logger_test.info(f"Test batch keys: {batch_test.keys()}")
    
    # --- Test training_step ---
    logger_test.info("Testing training_step...")
    try:
        loss_train = model_test.training_step(batch_test, 0)
        if loss_train is not None:
             logger_test.info(f"  Training step loss: {loss_train.item():.4f}")
        else:
             logger_test.warning("  Training step returned None (batch might have been skipped).")
    except Exception as e:
        logger_test.error(f"  Error during training_step: {e}", exc_info=True)

    # --- Test validation_step (if val data exists) ---
    val_dataloader = dm_test.val_dataloader()
    if val_dataloader:
        logger_test.info("Testing validation_step...")
        try:
            batch_val_test = next(iter(val_dataloader))
            if batch_val_test:
                loss_val = model_test.validation_step(batch_val_test, 0)
                if loss_val is not None:
                    logger_test.info(f"  Validation step loss: {loss_val.item():.4f}")
                else:
                    logger_test.warning("  Validation step returned None.")
            else:
                logger_test.warning("  Could not get a validation batch for testing.")
        except Exception as e:
            logger_test.error(f"  Error during validation_step: {e}", exc_info=True)
    else:
        logger_test.info("Skipping validation_step test as no validation dataloader.")

    # --- Test generation ---
    # For generation, we need to prepare appropriate inputs based on the task type tested
    logger_test.info("Testing generate method...")
    try:
        context_ids_gen, prompt_ids_gen = None, None
        context_mask_gen = None

        if data_cfg_test.task == "generative" and data_cfg_test.use_snippet:
            if "encoder_ids" in batch_test:
                context_ids_gen = batch_test["encoder_ids"] # (B, L_enc)
                context_mask_gen = batch_test.get("encoder_mask")
            if "decoder_ids" in batch_test: # Use start of decoder sequence as prompt
                prompt_ids_gen = batch_test["decoder_ids"][:, :5] # (B, L_prompt_short)
                # Or, more typically for generation, just a BOS token or a very short seed
                # prompt_ids_gen = torch.tensor([[model_test.bos_id]], device=model_test.device, dtype=torch.long)

        elif data_cfg_test.task == "generative" and not data_cfg_test.use_snippet:
             if "input_ids" in batch_test:
                context_ids_gen = batch_test["input_ids"] # Encoder sees full input
                context_mask_gen = batch_test.get("attention_mask")
                # For "full generative", prompt might be just BOS or a short prefix
                prompt_ids_gen = batch_test["input_ids"][:, :5] # Example prompt
                # prompt_ids_gen = torch.tensor([[model_test.bos_id]], device=model_test.device, dtype=torch.long)
        
        if prompt_ids_gen is None: # Default prompt if not set by task logic
            prompt_ids_gen = torch.tensor([[model_test.bos_id, 60, 70]], device=model_test.device, dtype=torch.long) # Dummy prompt: BOS, note, note
            logger_test.info(f"Using default prompt for generation: {prompt_ids_gen}")


        if context_ids_gen is not None and prompt_ids_gen is not None:
            logger_test.info(f"  Generating with context_ids shape: {context_ids_gen.shape}, prompt_ids shape: {prompt_ids_gen.shape}")
            generated_output_ids = model_test.generate(
                prompt_ids=prompt_ids_gen,
                context_ids=context_ids_gen,
                context_mask=context_mask_gen,
                max_new_tokens=20, # Generate few tokens for test
                temperature=0.7,
                output_path="./dummy_data/generated_test.mid", # Will create if path exists
                debug=False # Set to True for verbose generation logging
            )
            logger_test.info(f"  Generated sequence shape: {generated_output_ids.shape}")
            if os.path.exists("./dummy_data/generated_test.mid"):
                logger_test.info(f"  Generated output saved to ./dummy_data/generated_test.mid")
            
            # Decode a small part for verification
            if generated_output_ids.numel() > 0:
                decoded_text = dummy_tokenizer.decode(generated_output_ids[0].cpu().tolist())
                logger_test.info(f"  Decoded generated (first 30 tokens): {decoded_text[:100]}") # Print first 100 chars
        else:
            logger_test.warning("  Skipping generation test as context_ids or prompt_ids could not be prepared from the batch or task type.")

    except Exception as e:
        logger_test.error(f"  Error during generate method: {e}", exc_info=True)
    
    # Clean up dummy files
    # os.remove(dummy_train_path_pkl)
    # os.remove(dummy_val_path_pkl)
    # # Consider removing dummy_data directory if safe
    logger_test.info("StariaModel __main__ test block finished. (Dummy files not automatically cleaned up)")