from music import tokenize_midi_folder, train_model, generate_music
from miditok import REMI
from symusic import Score
from miditoolkit import MidiFile
import torch
import json

tokenizer = REMI()
#tokenize_midi_folder("/Users/ritik/Documents/GitHub/ai/midi_data", "/Users/ritik/Documents/GitHub/ai/tokens")
##for i in range(20):
    ##print(f"Iteration: {i}")
model = train_model("/home/ritikraman/ritik/gcp_ai/tokens", vocab_size=283, seq_len=512)

  # Adjust vocab_size after checking tokenizer.vocab

# Step 3: Generate music
seed = [4, 4, 51, 53]  # Can be anything — maybe the start of a real piece
new_tokens = generate_music(model, seed, max_len=500, temperature=0.8, top_k=50)
print(f"Generated {len(new_tokens)} tokens")
print(f"Token range: {min(new_tokens)} to {max(new_tokens)}")
print(f"First 20 tokens: {new_tokens[:20]}")

# Decode the generated tokens back to MIDI
print("\n=== DECODING GENERATED TOKENS ===")
score_tick = tokenizer.decode([new_tokens])
print(f"Decoded score type: {type(score_tick)}")
print(score_tick)


try:
    # Create a Score object and copy the data from ScoreTick
    score = Score()
    # Copy the tracks from ScoreTick to Score
    for track in score_tick.tracks:
        score.tracks.append(track)
    score.dump_midi("generated_output.mid")
    print("Saved generated_output.mid using Score conversion")
except Exception as e:
    print(f"Error with Score conversion: {e}")
