# generator_finetuner_pl.py
import logging, os, torch, torch.nn as nn, pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from src.StariaTokenizer import MusicTokenizerWithStyle, IGNORE_LABEL_IDX   # ← central constants
from src.utils import get_batch_prompts_from_midi_style_ids
from src.aria_generative.config import load_model_config
from src.aria_generative.model import ModelConfig, TransformerLM
from src.aria_generative.utils import _load_weight
from src.MidiClassifierModel import MidiClassifier

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class GeneratorFinetuner(pl.LightningModule):
    def __init__(self, 
                 style_vocab_size: int, 
                 max_seq_len: int,
                 learning_rate_gen: float = 1e-5,
                 generative_config_path: str = None,
                 generative_weights_path: str = None,
                 inference_checkpoint_path: str = None,
                 cpa_temp: float = 0.05,          # NEW ▶ temperature τ
                 cpa_lambda: float = 0.1,         # NEW ▶ weighting λ
                 cpa_k_head: int = 8,             # NEW ▶ #tokens for cont-repr
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # -------------- tokenizer / loss -----------------
        self.tokenizer = MusicTokenizerWithStyle()
        self.pad_token_id = self.tokenizer.pad_id
        self.vocab_size   = self.tokenizer.vocab_size
        self.criterion = nn.CrossEntropyLoss(
            ignore_index = self.pad_token_id, reduction='none')

        # -------------- models ---------------------------
        self.generative_model = None   # loaded in _load_generative_model
        self.inference_model  = None
        self._load_generative_model()
        self._load_inference_model()

        # -------------- CPA params -----------------------
        self.tau         = cpa_temp
        self.lambda_cpa  = cpa_lambda
        self.k_head      = cpa_k_head     # see §2.2 in the previous answer

    # ----------------------------------------------------
    def _load_generative_model(self):
        if self.hparams.generative_config_path:
            cfg_dict = load_model_config(self.hparams.generative_config_path)
            cfg_dict['vocab_size'] = self.vocab_size
            model_cfg = ModelConfig(**cfg_dict)
            self.generative_model = TransformerLM(model_cfg)
            if self.hparams.generative_weights_path:
                sd = _load_weight(self.hparams.generative_weights_path, device='cpu')
                sd = {k:v for k,v in sd.items() if 'rotary_emb' not in k}
                self.generative_model.load_state_dict(sd, strict=False)
        else:
            self.generative_model = TransformerLM(ModelConfig(vocab_size=self.vocab_size))

    def _load_inference_model(self):
        self.inference_model = MidiClassifier.load_from_checkpoint(
            self.hparams.inference_checkpoint_path, map_location='cpu')
        self.inference_model.eval()
        for p in self.inference_model.parameters(): p.requires_grad_(False)

    # ----------------------------------------------------
    def forward(self, ids): return self.generative_model(ids)
    
    def training_step(self, batch, batch_idx):
        """Wake Phase 2 (generator) optimisation step."""
        # 1. Get real tokens and infer latent style labels
        X_real = batch["input_ids"]
        
        with torch.no_grad():
            self.inference_model.eval()
            attention_mask = torch.ones_like(X_real)
            Z_inf_real = torch.argmax(self.inference_model(X_real, attention_mask), dim=-1)
            
            batch_prompts, _ = get_batch_prompts_from_midi_style_ids(
                input_tokens_batch=X_real,
                style_ids_batch=Z_inf_real,
                tokenizer=self.tokenizer,
                max_prompt_length=256,
            )
        
        # 2. Build sequences with prompts + real tokens + EOS
        eos_id, pad_id = self.tokenizer.eos_id, self.pad_token_id
        batch_full_sequences, prompt_lens = [], []
        
        for i, prompt in enumerate(batch_prompts):
            prompt_tensor = torch.tensor(prompt, device=self.device)
            x_real_no_pad = X_real[i][X_real[i] != pad_id]
            seq = torch.cat([prompt_tensor, x_real_no_pad])
            
            # Add EOS if needed
            if seq[-1] != eos_id:
                seq = torch.cat([seq, torch.tensor([eos_id], device=self.device)])
                
            batch_full_sequences.append(seq)
            prompt_lens.append(len(prompt_tensor))
        
        # 3. Pad sequences to max length + 1
        max_len = max(seq.size(0) for seq in batch_full_sequences) + 1
        padded = []
        for seq in batch_full_sequences:
            need = max_len - seq.size(0)
            pad = torch.full((need,), pad_id, device=self.device, dtype=seq.dtype)
            padded.append(torch.cat([seq, pad]))
        full_sequences = torch.stack(padded, dim=0)
        
        # 4. Create input/target pairs with 1-token shift
        input_tokens = full_sequences[:, :-1]
        target_tokens = full_sequences[:, 1:]
        
       # -------- forward *once* through transformer, grab hidden & logits ----
        hidden = self.generative_model.model(input_tokens)          # (B,S,D)
        logits = self.generative_model.lm_head(hidden)              # (B,S,V)

        

        # ------------------ 1. language-model loss (safe) ----------------
        lm_loss_tok = self.criterion(
            logits.reshape(-1, self.vocab_size),
            target_tokens.reshape(-1)
        ).view_as(target_tokens)

        mask = (target_tokens != self.pad_token_id)
        for i, P in enumerate(prompt_lens):
            if P > 1:
                mask[i, :P-1] = False  # exclude prompt tokens

        # sum up only non-pad, non-prompt tokens
        token_count = mask.sum()
        if token_count == 0:
            # no valid tokens this batch → skip LM loss
            lm_loss = torch.tensor(0.0, device=self.device)
        else:
            lm_loss = (lm_loss_tok * mask).sum() / token_count

        # ------------------ 2. contrastive-prompt alignment loss --------------
        reps_p, reps_c = [], []
        for i, P in enumerate(prompt_lens):
            p, c = self._get_prompt_cont_repr(hidden[i], P)
            reps_p.append(p); reps_c.append(c)
        reps_p = torch.stack(reps_p)           # (B,D)
        reps_c = torch.stack(reps_c)           # (B,D)

        # cosine similarities → logits for InfoNCE
        reps_p = F.normalize(reps_p, dim=-1)
        reps_c = F.normalize(reps_c, dim=-1)
        cpa_logits = reps_p @ reps_c.T / self.tau          # (B,B)

        targets = torch.arange(cpa_logits.size(0),
                               device=self.device, dtype=torch.long)
        cpa_loss = F.cross_entropy(cpa_logits, targets)

        # ------------------ 3. combine & log -----------------------------------
        total_loss = lm_loss + self.lambda_cpa * cpa_loss
        self.log_dict({"lm_loss": lm_loss,
                       "cpa_loss": cpa_loss,
                       "total_loss": total_loss},
                      on_step=True, prog_bar=True, sync_dist=True)
        return total_loss
    
    def _get_prompt_cont_repr(self, hidden: torch.Tensor,
                              prompt_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden      : (S, D)   final layer-norm’d states of ONE sample
        prompt_len  : length of the prompt (incl. structure tokens)
        returns p, c each (D,)
        """
        p = hidden[prompt_len - 1]                              # last prompt tok
        k = min(self.k_head, hidden.size(0) - prompt_len) or 1  # at least 1
        c = hidden[prompt_len : prompt_len + k].mean(dim=0)     # first k tokens
        return p, c

    def configure_optimizers(self):
        """Configure the optimizer for the generative model."""
        return torch.optim.AdamW(
            self.generative_model.parameters(),
            lr=self.hparams.learning_rate_gen
        )

    def validation_step(self, batch, batch_idx):
        """Validation step for the generative model, including contrastive prompt alignment."""
        with torch.no_grad():
            self.inference_model.eval()
            
            # 1. Get data and infer style labels
            X_real = batch['input_ids'].to(self.device)
            attention_mask = torch.ones_like(X_real)
            Z_inf_real = torch.argmax(self.inference_model(X_real, attention_mask), dim=-1)
            
            # 2. Create prompts
            batch_prompts, _ = get_batch_prompts_from_midi_style_ids(
                input_tokens_batch=X_real,
                style_ids_batch=Z_inf_real,
                tokenizer=self.tokenizer,
                max_prompt_length=256
            )
            
            # 3. Build sequences
            eos_id = self.tokenizer.eos_id
            pad_id = self.pad_token_id
            batch_full_sequences, prompt_lens = [], []
            
            for i, prompt in enumerate(batch_prompts):
                prompt_tensor = torch.tensor(prompt, device=self.device)
                prompt_lens.append(len(prompt_tensor))
                x_real_no_pad = X_real[i][X_real[i] != pad_id]
                seq = torch.cat([prompt_tensor, x_real_no_pad])
                
                if seq[-1] != eos_id:
                    seq = torch.cat([seq, torch.tensor([eos_id], device=self.device)])
                    
                batch_full_sequences.append(seq)
            
            # 4. Pad sequences
            max_len = max(len(seq) for seq in batch_full_sequences)
            padded = []
            for seq in batch_full_sequences:
                need = max_len - seq.size(0)
                pad = torch.full((need,), pad_id, device=self.device, dtype=seq.dtype)
                padded.append(torch.cat([seq, pad]))
            full_sequences = torch.stack(padded)
            
            # 5. Create input/target pairs
            input_tokens = full_sequences[:, :-1]
            target_tokens = full_sequences[:, 1:]
            
            # 6. Forward pass through transformer, get hidden and logits
            hidden = self.generative_model.model(input_tokens)          # (B,S,D)
            logits = self.generative_model.lm_head(hidden)              # (B,S,V)
            
            # 7. Compute language modeling loss with masking
            loss_per_tok = self.criterion(
                logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1)
            ).view_as(target_tokens)
            
            # Create mask for non-prompt tokens
            mask = (target_tokens != self.pad_token_id)
            for i, p_len in enumerate(prompt_lens):
                if p_len > 1:
                    mask[i, :p_len-1] = False  # exclude prompt tokens
            
            valid_token_count = mask.sum()
            if valid_token_count > 0:
                val_lm_loss = (loss_per_tok * mask).sum() / valid_token_count
            else:
                val_lm_loss = torch.tensor(0.0, device=self.device)
                if self.trainer.global_rank == 0:
                    logger.warning("Valid token count is zero in validation step, setting loss to 0.")
            
            # 8. Contrastive prompt alignment loss (CPA)
            reps_p, reps_c = [], []
            for i, P in enumerate(prompt_lens):
                p, c = self._get_prompt_cont_repr(hidden[i], P)
                reps_p.append(p)
                reps_c.append(c)
            reps_p = torch.stack(reps_p)           # (B,D)
            reps_c = torch.stack(reps_c)           # (B,D)
            reps_p = F.normalize(reps_p, dim=-1)
            reps_c = F.normalize(reps_c, dim=-1)
            cpa_logits = reps_p @ reps_c.T / self.tau          # (B,B)
            targets = torch.arange(cpa_logits.size(0), device=self.device, dtype=torch.long)
            cpa_loss = F.cross_entropy(cpa_logits, targets)
            
            # 9. Combine and log
            total_loss = val_lm_loss + self.lambda_cpa * cpa_loss
            self.log_dict({
                "val_lm_loss": val_lm_loss,
                "val_cpa_loss": cpa_loss,
                "val_total_loss": total_loss
            }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return total_loss
    @torch.inference_mode()
    def generate_from_prompt(
            self,
            prompt,                # List[int]: your encoded prompt tokens
            max_gen_tokens   = 512,
            temperature      = 1.0,
            top_p            = 0.95,
            repetition_window = 32,
            repetition_threshold = 3,
            force_end        = False
        ):
        """
        Same prompt structure as before, but:
        - top-p sampling,
        - light repetition penalty,
        - KV cache for efficient incremental decoding.
        Returns: (full_sequence_ids, generated_part_ids)
        """
        device = self.device
        tok    = self.tokenizer
        pad_id = tok.pad_id
        eos_id = tok.eos_id

        # ——————————————————————————————————————————————————————
        # 1) Build the initial sequence exactly as you did before
        prefix_ids = [
            tok.encode([('prefix','instrument','piano')])[0],
            tok.encode(['<S>'])[0],
        ]
        generated = list(prompt) + prefix_ids

        # 2) No KV-cache: just use full context at each step
        total_len = len(generated) + max_gen_tokens
        dtype     = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        from tqdm import tqdm

        # ——————————————————————————————————————————————————————
        # 3) Sampling loop
        def sample_top_p_logits(logits_1d):
            # logits_1d: (V,)
            logits_1d = logits_1d.float()  # ensure float for softmax/multinomial
            probs = torch.softmax(logits_1d / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            # keep tokens up to where cumulative prob exceeds top_p
            cutoff = (cumprobs > top_p).nonzero(as_tuple=True)[0]
            if len(cutoff) > 0:
                last = cutoff[0].item() + 1
                sorted_probs[last:] = 0.0
            # renormalize
            if sorted_probs.sum() == 0:
                sorted_probs = torch.ones_like(sorted_probs) / sorted_probs.size(0)
            else:
                sorted_probs.div_(sorted_probs.sum())
            choice = torch.multinomial(sorted_probs, 1)
            return sorted_indices[choice].item()

        for step in tqdm(range(max_gen_tokens), desc="Generating", leave=False):
            # Feed the entire sequence so far (no KV cache)
            inp = torch.tensor(generated, device=device).unsqueeze(0)
            logits = self.generative_model(inp)[:, -1, :]  # shape (1, V) → take [0]

            l1 = logits[0]

            # — light repetition penalty on recent tokens
            if repetition_window > 0:
                recent = generated[-repetition_window:]
                counts = {}
                for t in recent:
                    counts[t] = counts.get(t, 0) + 1
                for t, c in counts.items():
                    if c >= repetition_threshold:
                        l1[t] = l1[t] / (c + 1)

            # No more dim_tok/force_end logic
            next_id = sample_top_p_logits(l1)

            generated.append(next_id)
            if next_id == eos_id:
                break

        # ——————————————————————————————————————————————————————
        # 4) Return full + “no prompt” portion
        music_start = len(prompt) 
        output_no_prompt = [t for t in generated[music_start:] if t != pad_id]
        return generated, output_no_prompt


    def get_next_token_probs(self, seq):
        """Get top 5 probabilities and tokens for the next token."""
        self.eval()
        with torch.no_grad():
            if isinstance(seq, list):
                input_tensor = torch.tensor(seq, device=self.device).unsqueeze(0)
            else:
                input_tensor = seq.unsqueeze(0) if seq.dim() == 1 else seq.to(self.device)
                
            logits = self(input_tensor)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            return top_probs, top_indices.tolist()