# music_llm.py

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from miditok import REMI
from miditoolkit import MidiFile
from tqdm import tqdm

# --- Tokenizer ---
def tokenize_midi_folder(midi_folder, output_folder):
    tokenizer = REMI()
    midi_paths = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith('.mid')]
    for midi_path in midi_paths:
        print(f"Tokenizing {midi_path}")
        try:
            midi = MidiFile(midi_path)
            tokens = tokenizer(midi)
            output_path = os.path.join(output_folder, os.path.basename(midi_path) + ".json")
            tokenizer.save_tokens(tokens, output_path)
        except Exception as e:
            print(f"Error tokenizing {midi_path}: {e}")
            try:
                os.remove(midi_path)
                print(f"Deleted problematic file: {midi_path}")
            except Exception as delete_error:
                print(f"Failed to delete {midi_path}: {delete_error}")
# --- Dataset ---

class MusicDataset(Dataset):
    def __init__(self, token_folder, seq_len=512, max_files=None):
        self.data = []
        self.seq_len = seq_len

        # Get all available .json files
        all_fnames = [f for f in os.listdir(token_folder) if f.endswith('.json')]

        # Sample a subset of files each time
        if max_files:
            fnames = random.sample(all_fnames, min(max_files, len(all_fnames)))
        else:
            fnames = all_fnames

        # Load tokens from selected files
        for fname in fnames:
            with open(os.path.join(token_folder, fname)) as f:
                token_data = json.load(f)
                file_tokens = token_data['ids']
                for track_tokens in file_tokens:
                    # Skip very short tracks
                    if len(track_tokens) > seq_len:
                        for i in range(0, len(track_tokens) - seq_len):
                            seq = track_tokens[i:i + seq_len + 1]
                            self.data.append(seq)
                print(f"Loaded {len(file_tokens)} tracks from {fname}, total tokens: {sum(len(t) for t in file_tokens)}")

        #print(f"Total usable sequences in dataset: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)
# --- Model ---
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1)]
        x = self.transformer(x)
        return self.fc_out(x)

# --- Training ---
def get_all_token_files(token_folder):
    return [os.path.join(token_folder, f) for f in os.listdir(token_folder) if f.endswith('.json')]

def chunkify(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def train_model(token_folder, vocab_size=512, epochs=113, batch_size=16, seq_len=512):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    all_files = get_all_token_files(token_folder)
    random.shuffle(all_files)
    file_chunks = chunkify(all_files, 50)

    model = MusicTransformer(vocab_size, d_model=256, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if os.path.exists("512_music_model_checkpoint.pt"):
        checkpoint = torch.load("512_music_model_checkpoint.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model from epoch {start_epoch}")

    epoch_counter = start_epoch
    for epoch in range(start_epoch, start_epoch + epochs):
        for chunk_idx, file_chunk in enumerate(file_chunks):
            print(f"\nEpoch {epoch_counter+1} | Chunk {chunk_idx+1}/{len(file_chunks)}")

            dataset = MusicDataset(None, seq_len=seq_len, max_files=None)
            dataset.data = []  # Reset data manually
            for fname in file_chunk:
                
                with open(fname) as f:
                    token_data = json.load(f)
                    file_tokens = token_data['ids']
                    for track_tokens in file_tokens:
                        if len(track_tokens) > seq_len:
                            for i in range(0, len(track_tokens) - seq_len):
                                seq = track_tokens[i:i + seq_len + 1]
                                dataset.data.append(seq)
                print(f"Loaded {len(file_tokens)} tracks from {fname}, total tokens: {sum(len(t) for t in file_tokens)}")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print(f"Total usable sequences in dataset: {len(dataset.data)}")

            model.train()
            total_loss = 0
            for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch_counter+1}"):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            checkpoint = {
                'epoch': epoch_counter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, "512_music_model_checkpoint.pt")
            print(f"Epoch {epoch_counter+1} complete | Avg Loss: {avg_loss:.4f}")

            with open("loss_history.txt", "a") as f:
                f.write(f"{epoch_counter+1},{avg_loss:.4f}\n")

            epoch_counter += 1

    return model

# --- Generation ---
def generate_music(model, start_seq, max_len=1024, temperature=1.0, top_k=50, window=512):

    model.eval()
    device = next(model.parameters()).device
    context_window = window

    # Initialize sequence
    seq = torch.tensor(start_seq, dtype=torch.long).unsqueeze(0).to(device)
    print(f"Generating music with max_len={max_len}, temperature={temperature}, top_k={top_k}")

    sincebar = 0  # allow token 4 right away
    i = 0

    while i < max_len:
        # Truncate sequence to last 20 tokens (fixed context window)
        input_seq = seq[:, -context_window:]
        with torch.no_grad():
            logits = model(input_seq)[:, -1, :] / temperature

            # If 4 is not allowed yet, mask it out
            if sincebar < 30:
                logits[0, 4] = float('-inf')  # block token 4

            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, next_token_idx)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_val = next_token.item()
        seq = torch.cat((seq, next_token), dim=1)

        # Update sincebar logic
        sincebar = 0 if token_val == 4 else sincebar + 1

        print(token_val)
        i += 1
        if i % 100 == 0:
            print(f"Generated {i}/{max_len} tokens...")

    return seq.squeeze().tolist()
