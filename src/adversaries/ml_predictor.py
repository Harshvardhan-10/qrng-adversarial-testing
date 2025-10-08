"""Train a small LSTM to predict next bit from previous k bits."""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits

def train(bits, epochs=10, window=16, batch=64):
    dataset = BitDataset(bits, window)
    dl = DataLoader(dataset, batch_size=batch, shuffle=True)
    model = LSTMPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        tot_loss = 0.0
        correct = 0
        total = 0
    
        for X, y in dl:
            X = X
            y = y
            logits = model(X)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch}: loss={tot_loss/len(dl):.4f} acc={correct/total:.4f}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/qrng_1k.npz')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    arr = np.load(args.data)['bits']
    train(arr, epochs=args.epochs)