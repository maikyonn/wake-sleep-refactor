from GeneratorModule import GeneratorFinetuner
from utils import get_prompt_from_midi_snippets
from tokenizer import MusicTokenizerWithStyle
import torch

# Initialize the tokenizer
tokenizer = MusicTokenizerWithStyle()

# Load and tokenize the MIDI file
midi_path = "datasets/10k-various-synth-test-set/midi/ABA_143.mid"
midi_snippet = tokenizer.tokenize_from_file(midi_path)
if midi_snippet is None:
    raise ValueError(f"Failed to tokenize MIDI file: {midi_path}")

print(len(midi_snippet))
if len(midi_snippet) > 3840:
    print("MIDI snippet is too long")
    exit()

tokenizer.ids_to_file(tokenizer.encode(midi_snippet), "gt-output.mid")

# Specify the checkpoint path you want to load
checkpoint_path = "checkpoints/genfinetune_bs16_lr1e-05_flim200000_seq3820_gacc4_node2300_20250503_164306/genfinetune-epoch=02-step=420-val_total_loss=1.0428.ckpt"

# Initialize the generator from checkpoint
gen = GeneratorFinetuner.load_from_checkpoint(checkpoint_path).to("cuda")

# Encode the tokenized MIDI to input IDs
input_ids = torch.tensor(tokenizer.encode(midi_snippet)).unsqueeze(0).to(gen.device)  # [1, seq_len]

# Prepare attention mask (all ones, since no padding for single example)
attention_mask = torch.ones_like(input_ids)

# Run the classification (inference) model to get style predictions
with torch.no_grad():
    # gen.inference_model expects (input_ids, attention_mask)
    inf_logits = gen.inference_model(input_ids, attention_mask)  # [1, seq_len, style_vocab_size]
    style_ids = torch.argmax(inf_logits, dim=-1)  # [1, seq_len]

# Create prompts using the same function as in training
from utils import get_batch_prompts_from_midi_style_ids, filter_significant_styles, get_music_style_from_condensed_style_sequence

style_ids_list = style_ids[0].tolist()
style_tokens = [tokenizer.idx_to_style[idx] for idx in style_ids_list]

# Filter significant styles and get music style
style_seqs = filter_significant_styles(style_tokens)
music_style = get_music_style_from_condensed_style_sequence(style_seqs)
print(music_style)

batch_prompts, batch_prompt_tokens = get_batch_prompts_from_midi_style_ids(
    input_tokens_batch=input_ids,
    style_ids_batch=style_ids,
    tokenizer=tokenizer,
    max_prompt_length=256
)

# For single example, take the first prompt
prompt = batch_prompts[0]

full_ids, gen_ids = gen.generate_from_prompt(prompt,
                                             max_gen_tokens=3700,
                                             temperature=0.9,
                                             top_p=0.95,
                                             force_end=True)
print(gen_ids)
print(tokenizer.decode(gen_ids))


tokenizer.ids_to_file(gen_ids, "output.mid")
