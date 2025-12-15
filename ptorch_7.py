import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from multiprocessing import freeze_support
from time import time
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# --------------------------------------------------
# CUSTOM DATASET ZA LOTO
# --------------------------------------------------
class LotoDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file).values.astype('float32')  # shape [N,7]
        self.x = data[:-1]  # sve osim poslednje
        self.y = data[1:]   # sve osim prve
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


# --------------------------------------------------
# ---------- Standard Training (Baseline) ----------
# MLP MODEL
# --------------------------------------------------
class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_dim=7, output_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# --------------------------------------------------
# ------------ Multi-Worker Data Loading -----------
# RNN MODEL
# --------------------------------------------------
class RNNModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=7):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # poslednji output sekvence
        return self.fc(out)


# --------------------------------------------------
# ------ Pinned Memory + Non-blocking Transfer -----
# LSTM MODEL
# --------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # poslednji output sekvence
        return self.fc(out)


# --------------------------------------------------
# TRAIN LOOP
# --------------------------------------------------
def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0

    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)

        # za RNN/LSTM dodaj dimenziju sekvence
        if len(data.shape) == 2:
            data_seq = data.unsqueeze(1)  # (batch, seq_len=1, input_dim)
        else:
            data_seq = data

        optimizer.zero_grad()
        output = model(data_seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Avg loss: {total_loss/len(dataloader):.4f}")


# --------------------------------------------------
# FUNKCIJA ZA PREDIKCIJU SLEDEĆE LOTO KOMBINACIJE
# --------------------------------------------------
def predict_next_combination(model, dataset, device):
    model.eval()
    last_comb = dataset[-1][0]
    if isinstance(last_comb, (pd.Series, np.ndarray)):
        last_comb = torch.tensor(last_comb, dtype=torch.float32)

    x = last_comb.unsqueeze(0).unsqueeze(1).to(device)  # batch + seq dim
    with torch.no_grad():
        pred = model(x)

    # Ograniči vrednosti između 1 i 39
    pred_clamped = torch.clamp(pred, 1, 39)

    # Zaokruži na ceo broj i konvertuj u listu
    pred_rounded = [int(torch.round(p)) for p in pred_clamped.squeeze(0)]
    return pred_rounded


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("STUDY OF ML TRAINING OPTIMIZATION")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Training Device: {device}")
    print("=" * 60)
    print()

    train_dataset = LotoDataset("/Users/milan/Desktop/GHQ/data/loto7_4528_k98.csv")

    # ---------------- MLP ----------------
    model_mlp = SimpleFeedForwardNN().to(device)
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=3, persistent_workers=True)

    print("\n================ MLP (Feedforward) TRAINING ================\n")
    start = time()
    for epoch in range(1, 1001):
        train(model_mlp, train_loader, optimizer, criterion, device, epoch)
    print(f"Time: {time() - start:.2f}s\n")

    # ---------------- RNN ----------------
    model_rnn = RNNModel().to(device)
    optimizer = optim.Adam(model_rnn.parameters(), lr=0.001)
    train_loader_rnn = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=3, persistent_workers=True)

    print("\n================ RNN TRAINING ================\n")
    start = time()
    for epoch in range(1, 1001):
        train(model_rnn, train_loader_rnn, optimizer, criterion, device, epoch)
    print(f"Time: {time() - start:.2f}s\n")

    # ---------------- LSTM ----------------
    model_lstm = LSTMModel().to(device)
    optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)
    train_loader_lstm = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=3, persistent_workers=True)

    print("\n================ LSTM TRAINING ================\n")
    start = time()
    for epoch in range(1, 1001):
        train(model_lstm, train_loader_lstm, optimizer, criterion, device, epoch)
    print(f"Time: {time() - start:.2f}s\n")

    # Predikcija sledeće loto kombinacije
    next_loto_mlp = predict_next_combination(model_mlp, train_dataset, device)
    next_loto_rnn = predict_next_combination(model_rnn, train_dataset, device)
    next_loto_lstm = predict_next_combination(model_lstm, train_dataset, device)

    print("\n================ PREDIKCIJE SLEDEĆE LOTO KOMBINACIJE ================\n")
    print("Predikcija (MLP):", next_loto_mlp)
    print("Predikcija (RNN):", next_loto_rnn)
    print("Predikcija (LSTM):", next_loto_lstm)
    print()
    print("✅ ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    print()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    freeze_support()
    main()



"""
...
Epoch 995 | Avg loss: 26.6622
Epoch 996 | Avg loss: 26.6741
Epoch 997 | Avg loss: 26.6983
Epoch 998 | Avg loss: 26.7097
Epoch 999 | Avg loss: 26.6525
Epoch 1000 | Avg loss: 26.6497
Time: 49.30s



Epoch 1000
================ PREDIKCIJE SLEDEĆE LOTO KOMBINACIJE ================

Predikcija (MLP): [6, 11, 16, 20, 24, 29, 34]
Predikcija (RNN): [5, 10, 15, 20, 25, 30, 35]
Predikcija (LSTM): [5, 10, 16, 21, 26, 31, 35]

✅ ALL EXPERIMENTS COMPLETED
============================================================
"""



"""
============================================================
STUDY OF ML TRAINING OPTIMIZATION
============================================================
PyTorch Version: 2.8.0
Training Device: cpu
============================================================
"""




