import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import argparse

# -------------------------------
# Dataset: sliding windows over bits
# -------------------------------
class BitDataset(Dataset):
    def __init__(self, bits, window=16):
        self.X = []
        self.y = []
        for i in range(len(bits) - window):
            self.X.append(bits[i:i+window])
            self.y.append(bits[i+window])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------
# LSTM model
# -------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(-1)        # -> (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]        # last timestep
        logits = self.fc(out)
        return logits

# -------------------------------
# Evaluate function (for validation)
# -------------------------------
def evaluate(model, dl, loss_fn):
    model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dl:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return tot_loss / len(dl), correct / total

# -------------------------------
# Train function
# -------------------------------
def train(bits, epochs=10, window=64, batch=256):
    print(f"Preparing dataset with window={window} ...")
    dataset = BitDataset(bits, window)

    # Split 90% train, 10% validation (keep order!)
    split = int(0.9 * len(dataset))
    train_ds = Subset(dataset, range(split))
    val_ds = Subset(dataset, range(split, len(dataset)))

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)

    model = LSTMPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {len(train_ds):,} samples, validating on {len(val_ds):,} samples")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0

        for X, y in train_dl:
            logits = model(X)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = tot_loss / len(train_dl)
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_dl, loss_fn)

        print(f"Epoch {epoch:2d}: "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    print("-" * 60)
    print("Training complete.")
    print("If val_acc ≈ 0.50 → generator is unpredictable (good).")
    print("If val_acc significantly > 0.50 → model found structure (bad for randomness).")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/prng_1M.npz",
                        help="Path to NPZ file containing 'bits' array of 0/1 values")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    arr = np.load(args.data)["bits"].astype(np.float32)
    train(arr, epochs=args.epochs, window=args.window, batch=args.batch)
