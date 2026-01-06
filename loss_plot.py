import matplotlib.pyplot as plt

epochs = []
losses = []

with open("loss_history.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                epoch = int(parts[0].strip())
                loss = float(parts[1].strip())
                epochs.append(epoch)
                losses.append(loss)
            except ValueError:
                print(f"Skipping malformed line: {line}")
        else:
            print(f"Skipping malformed line: {line}")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
