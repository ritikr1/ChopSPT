from music import MusicTransformer
import torch
from music import generate_music
from miditok import REMI
from symusic import Score
from miditoolkit import MidiFile, Instrument, Note

tokenizer = REMI()

# Recreate the model with the same architecture
model = MusicTransformer(vocab_size=283, d_model=256, num_layers=2)  # match your earlier settings
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load("512_music_model_checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

seed = [60]
new_tokens = generate_music(model, seed, max_len=500, temperature=0.2, top_k=10)
print(new_tokens)
score_tick = tokenizer.decode([new_tokens])
print(f"Decoded score type: {type(score_tick)}")
print(score_tick)

# The ScoreTick object should be converted to a Score object properly
# Let's try the correct way to create a Score from ScoreTick
try:
    # Create a Score object and copy the data from ScoreTick
    #score = Score()
    # Copy the tracks from ScoreTick to Score
    #for track in score_tick.tracks:
    #    score.tracks.append(track)
    score_tick.dump_midi("generated_output.mid")
    print("Saved generated_output.mid using Score conversion")
except Exception as e:
    print(f"Error with Score conversion: {e}")
    # Fallback: create a simple MIDI file manually
    from miditoolkit import MidiFile, Instrument, Note
    midi = MidiFile()
    # Create an instrument
    instrument = Instrument(program=0, is_drum=False)
    # Add a simple note to make it non-empty
    note = Note(velocity=64, pitch=60, start=0, end=1)
    instrument.notes.append(note)
    midi.instruments.append(instrument)
    midi.dump("generated_output.mid")
    print("Saved minimal generated_output.mid")


    