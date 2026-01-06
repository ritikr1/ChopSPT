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


    