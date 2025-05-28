from src.StariaModule import StariaModel, DecoderLM, ContrastiveStructureEncoder
from pytorch_lightning import Trainer
from src.MidiDataModule import DataCfg, MidiDataModule
from src.StariaTokenizer import MusicTokenizerWithStyle

tokenizer = MusicTokenizerWithStyle()

data_cfg = DataCfg(
    data_dir   = "cache/synthetic_aria-midi-v1-pruned-ext-200k-struct-v2_max4096_limit100_e3517256.pkl",
    mode       = "synthetic",
    task       = "generative",
    max_len    = 4096,
    seq_limit  = 1000,
    shuffle    = True,
    skip_long  = True,
    val_split  = 0.1
)
dm = MidiDataModule(
    cfg         = data_cfg,
    tokenizer   = tokenizer,
    batch_size  = 1,
    num_workers = 4,
    drop_last   = False,
) 

# Set up the data module
dm.setup()

# Get a sample batch from the training dataloader
train_dataloader = dm.train_dataloader()
sample_batch = next(iter(train_dataloader))

# Print the sample batch structure and content
print("Sample batch structure:")
for key, value in sample_batch.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {type(value)}")


model = StariaModel.from_pretrained(tokenizer, encoder_ckpt=None, decoder_ckpt="checkpoints/midi_decoder/epoch=00-train_loss=3.4266.ckpt")
trainer = Trainer(max_epochs=1)
trainer.fit(model, dm)
model.save_pretrained("staria-init/init.ckpt")

