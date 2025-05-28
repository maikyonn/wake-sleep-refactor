# tokenizer.py
# ---------------------------------------------------------------------
#  Global constants – import these elsewhere instead of redefining.
# ---------------------------------------------------------------------
IGNORE_LABEL_IDX   = -100     # Loss padding
STYLE_LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_STYLE_MAP    = {v: k for k, v in STYLE_LABEL_MAP.items()}

# Section / prompt tokens (strings only – IDs are looked-up dynamically)
PROMPT_START_TOKEN = '<PROMPT_START>'
PROMPT_END_TOKEN   = '<PROMPT_END>'
A_SECTION_TOKEN    = '<A_SECTION>'
B_SECTION_TOKEN    = '<B_SECTION>'
C_SECTION_TOKEN    = '<C_SECTION>'
D_SECTION_TOKEN    = '<D_SECTION>'

# List of all special tokens for easy reference
SPECIAL_TOKENS = [
    PROMPT_START_TOKEN,
    PROMPT_END_TOKEN,
    A_SECTION_TOKEN,
    B_SECTION_TOKEN,
    C_SECTION_TOKEN,
    D_SECTION_TOKEN,
]

# ----------------------------------------------
#  Implementation
# -----------------------------------------------------------------------------
import logging
from typing import List, Optional, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from external.ariautils.tokenizer.absolute import AbsTokenizer
from external.ariautils.midi import MidiDict

logger = logging.getLogger(__name__)

class MusicTokenizerWithStyle:
    """
    Thin wrapper around ariautils.AbsTokenizer providing:
      • Four style section indicators (A,B,C,D)
      • Prompt start/end markers
      • A library of structure-form tokens (<AB>, <ABA>, ...)
    All new tokens are appended to the underlying tokenizer's vocabulary.
    """

    # Allowable structure forms (will become tokens <A>, <AB>, <ABA>, …)
    STRUCTURE_FORMS = [
        "A", "AB", "ABC", "ABA", "ABAB", "ABAC", "ABCA", "ABCB", "ABCD"
    ]

    def __init__(self):
        # Initialize underlying ABS tokenizer
        self._tokenizer = AbsTokenizer(config_path="external/ariautils/config/config.json")

        # Register special tokens once
        special_tokens = [
            A_SECTION_TOKEN, B_SECTION_TOKEN, C_SECTION_TOKEN, D_SECTION_TOKEN,
            PROMPT_START_TOKEN, PROMPT_END_TOKEN,
            *[f"<{form}>" for form in self.STRUCTURE_FORMS]
        ]
        self._tokenizer.add_tokens_to_vocab(special_tokens)

        # Expose style maps
        self.idx_to_style = ID_TO_STYLE_MAP    # {0:"A",1:"B",...}
        self.style_to_idx = STYLE_LABEL_MAP    # {"A":0,...}

        print(
            f"MusicTokenizerWithStyle initialized | vocab_size={self.vocab_size} pad_id={self.pad_id} bos={self.bos_id} eos={self.eos_id}"
        )

    # --------------------------------------
    #  Public API
    # --------------------------------------
    def tokenize(self, midi_dict: MidiDict) -> List[str]:
        return self._tokenizer.tokenize(midi_dict)

    def encode(self, seq: List[str]) -> List[int]:
        return self._tokenizer.encode(seq)

    def decode(self, seq: List[int]) -> List[str]:
        return self._tokenizer.decode(seq)

    def tokenize_from_file(self, midi_path: str) -> Optional[List[str]]:
        try:
            midi_dict = MidiDict.from_midi(midi_path)
            return self.tokenize(midi_dict) or None
        except Exception as exc:
            logger.error(f"Tokenisation failed for {midi_path}: {exc}")
            return None

    def calc_length_ms(self, tokens: List[str], onset: bool = True) -> int:
        return self._tokenizer.calc_length_ms(tokens, onset)

    def truncate_by_time(self, tokens: List[str], trunc_time_ms: int) -> List[str]:
        return self._tokenizer.truncate_by_time(tokens, trunc_time_ms)

    def ids_to_file(self, ids: List[int], output_path: str):
        """Converts a list of integer IDs back into a MIDI file."""
        try:
            token_seq = self.decode(ids)
            midi_dict = self._tokenizer.detokenize(token_seq)
            midi = midi_dict.to_midi()
            midi.save(output_path)
            logger.info(f"Sample MIDI file saved to {output_path}")
        except Exception as e:
            logger.error(f"Error converting IDs to MIDI file: {e}", exc_info=False)

    def remove_instrument_prefix(self, tokens: List[str]) -> List[str]:
        """
        Removes instrument prefix tokens from a token sequence.
        
        The instrument prefix tokens correspond to ids 6 and 0 in the vocabulary.
        
        Args:
            tokens: List of string tokens
            
        Returns:
            List of tokens with instrument prefix tokens removed
        """
        # Get the actual token strings for ids 6 and 0
        instrument_tokens = []
        if 6 in self.id_to_tok:
            instrument_tokens.append(self.id_to_tok[6])
        if 0 in self.id_to_tok:
            instrument_tokens.append(self.id_to_tok[0])
            
        # Filter out the instrument tokens
        return [tok for tok in tokens if tok not in instrument_tokens]

    # --------------------------------------
    #  Properties
    # --------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def pad_id(self) -> int:
        return self._tokenizer.pad_id

    @property
    def bos_id(self) -> Optional[int]:
        return self._tokenizer.tok_to_id.get(self._tokenizer.bos_tok)

    @property
    def eos_id(self) -> Optional[int]:
        return self._tokenizer.tok_to_id.get(self._tokenizer.eos_tok)

    @property
    def tok_to_id(self) -> Dict[str,int]:
        return self._tokenizer.tok_to_id

    @property
    def id_to_tok(self) -> Dict[int,str]:
        return self._tokenizer.id_to_tok



# -------------------------------# ---------------------------------------------------------------------
# Form‐classification constants
# ---------------------------------------------------------------------
# Map each form‐string ("A","ABA","ABACA",…) to a class index
FORM_LABEL_MAP = {form: idx for idx, form in enumerate(MusicTokenizerWithStyle.STRUCTURE_FORMS)}
ID_TO_FORM_MAP = {idx: form for form, idx in FORM_LABEL_MAP.items()}