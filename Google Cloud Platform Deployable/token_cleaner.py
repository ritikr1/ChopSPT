import os
import json

tokens_dir = "/home/ritik/gcp_ai/tokens"
vocab_size = 283

bad_files = []

for fname in os.listdir(tokens_dir):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(tokens_dir, fname)
    try:
        with open(fpath, "r") as f:
            data = json.load(f)
        tokens = data.get("ids", [[]])[0]  # safely get first list of ids

        if not all(isinstance(t, int) for t in tokens):
            print(f"Non-integer token in {fname}")
            bad_files.append(fname)
        elif not all(0 <= t < vocab_size for t in tokens):
            print(f"Out-of-range token in {fname}")
            max_token = max(tokens)
            print(f"   Max token: {max_token}")
            bad_files.append(fname)

    except Exception as e:
        print(f"💥 Failed to process {fname}: {e}")
        bad_files.append(fname)

print(f"\nFinished scanning {len(os.listdir(tokens_dir))} files.")
print(f"Found {len(bad_files)} bad files.")
if bad_files:
    print("Examples:", bad_files[:5])
