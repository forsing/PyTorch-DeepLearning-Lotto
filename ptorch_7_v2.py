import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from multiprocessing import freeze_support
from time import time
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# v2: reproduktivnost
_TORCH_SEED = 39
torch.manual_seed(_TORCH_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_TORCH_SEED)


# --------------------------------------------------
# CUSTOM DATASET ZA LOTO
# --------------------------------------------------
class LotoDataset(Dataset):
    def __init__(self, csv_file):
        # v2: CSV bez zaglavlja (kao ostali loto fajlovi u GHQ)
        data = pd.read_csv(csv_file, header=None).values.astype("float32")  # shape [N,7]
        self._raw = data
        self.x = data[:-1]  # sve osim poslednje
        self.y = data[1:]   # sve osim prve
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

    def get_input_for_next_prediction(self):
        """Poslednji red u CSV = poslednje izvlačenje; ulaz za predikciju sledećeg."""
        return torch.tensor(self._raw[-1], dtype=torch.float32)


# --------------------------------------------------
# ---------- Standard Training (Baseline) ----------
# MLP MODEL
# --------------------------------------------------
class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_dim=7, output_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        return self.fc3(x)


# --------------------------------------------------
# ------------ Multi-Worker Data Loading -----------
# RNN MODEL
# --------------------------------------------------
class RNNModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=7, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
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
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=7, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # poslednji output sekvence
        return self.fc(out)


# --------------------------------------------------
# TRAIN LOOP
# --------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
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
    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            if len(data.shape) == 2:
                data_seq = data.unsqueeze(1)
            else:
                data_seq = data
            output = model(data_seq)
            total_loss += criterion(output, target).item()
    return total_loss / max(len(dataloader), 1)


# --------------------------------------------------
# FUNKCIJA ZA PREDIKCIJU SLEDEĆE LOTO KOMBINACIJE
# --------------------------------------------------
def predict_next_combination(model, dataset, device):
    model.eval()
    last_comb = dataset.get_input_for_next_prediction()
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

    csv_path = "/Users/4c/Desktop/GHQ/data/loto7_4586_k24.csv"
    full_dataset = LotoDataset(csv_path)
    n = len(full_dataset)
    split = int(0.85 * n)
    split = max(1, min(split, n - 1))
    train_ds = Subset(full_dataset, range(split))
    val_ds = Subset(full_dataset, range(split, n))

    # v2: vremenski red — bez shuffle; manje radnika da izbegnemo probleme na macOS
    _nw = 0
    train_loader_base = dict(
        batch_size=48,
        shuffle=False,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.MSELoss()
    max_epochs = 400
    log_every = 20
    patience = 35

    def run_training(name, model, train_loader, val_loader):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )
        best_val = float("inf")
        best_state = None
        stale = 0
        start = time()
        for epoch in range(1, max_epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"Epoch {epoch} | train {tr_loss:.4f} | val {val_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
                )
            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stop @ {epoch} (best val {best_val:.4f})")
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"Time ({name}): {time() - start:.2f}s\n")
        return model

    val_loader_common = DataLoader(val_ds, batch_size=48, shuffle=False, num_workers=_nw)

    # ---------------- MLP ----------------
    model_mlp = SimpleFeedForwardNN().to(device)
    train_loader = DataLoader(train_ds, **train_loader_base)

    print("\n================ MLP (Feedforward) TRAINING ================\n")
    model_mlp = run_training("MLP", model_mlp, train_loader, val_loader_common)

    # ---------------- RNN ----------------
    model_rnn = RNNModel().to(device)
    train_loader_rnn = DataLoader(train_ds, **train_loader_base)

    print("\n================ RNN TRAINING ================\n")
    model_rnn = run_training("RNN", model_rnn, train_loader_rnn, val_loader_common)

    # ---------------- LSTM ----------------
    model_lstm = LSTMModel().to(device)
    train_loader_lstm = DataLoader(train_ds, **train_loader_base)

    print("\n================ LSTM TRAINING ================\n")
    model_lstm = run_training("LSTM", model_lstm, train_loader_lstm, val_loader_common)

    # Predikcija sledeće loto kombinacije
    next_loto_mlp = predict_next_combination(model_mlp, full_dataset, device)
    next_loto_rnn = predict_next_combination(model_rnn, full_dataset, device)
    next_loto_lstm = predict_next_combination(model_lstm, full_dataset, device)

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
============================================================
STUDY OF ML TRAINING OPTIMIZATION
============================================================
PyTorch Version: 2.8.0
Training Device: cpu
============================================================


================ MLP (Feedforward) TRAINING ================

Epoch 1 | train 78.2468 | val 32.9281 | lr 1.00e-03
Epoch 20 | train 34.4933 | val 31.0061 | lr 1.00e-03
Epoch 40 | train 31.9932 | val 28.7771 | lr 1.00e-03
Epoch 60 | train 30.6126 | val 28.0525 | lr 1.00e-03
Epoch 80 | train 29.6472 | val 27.7812 | lr 1.00e-03
Epoch 100 | train 28.6297 | val 27.5530 | lr 2.50e-04
Epoch 120 | train 28.9099 | val 27.6301 | lr 1.25e-04
Epoch 140 | train 28.7534 | val 27.6890 | lr 3.13e-05
Early stop @ 151 (best val 27.5451)
Time (MLP): 9.17s


================ RNN TRAINING ================

Epoch 1 | train 315.6429 | val 185.5287 | lr 1.00e-03
Epoch 20 | train 26.9296 | val 27.4880 | lr 5.00e-04
Epoch 40 | train 26.9194 | val 27.4712 | lr 1.25e-04
Epoch 60 | train 26.9151 | val 27.4657 | lr 3.13e-05
Epoch 80 | train 26.9134 | val 27.4637 | lr 7.81e-06
Epoch 100 | train 26.9133 | val 27.4631 | lr 1.95e-06
Epoch 120 | train 26.9136 | val 27.4630 | lr 1.00e-06
Epoch 140 | train 26.9134 | val 27.4629 | lr 1.00e-06
Epoch 160 | train 26.9132 | val 27.4628 | lr 1.00e-06
Epoch 180 | train 26.9127 | val 27.4628 | lr 1.00e-06
Epoch 200 | train 26.9128 | val 27.4627 | lr 1.00e-06
Epoch 220 | train 26.9128 | val 27.4627 | lr 1.00e-06
Epoch 240 | train 26.9131 | val 27.4626 | lr 1.00e-06
Epoch 260 | train 26.9128 | val 27.4626 | lr 1.00e-06
Epoch 280 | train 26.9135 | val 27.4626 | lr 1.00e-06
Epoch 300 | train 26.9132 | val 27.4626 | lr 1.00e-06
Epoch 320 | train 26.9129 | val 27.4626 | lr 1.00e-06
Epoch 340 | train 26.9130 | val 27.4626 | lr 1.00e-06
Epoch 360 | train 26.9132 | val 27.4626 | lr 1.00e-06
Epoch 380 | train 26.9129 | val 27.4626 | lr 1.00e-06
Epoch 400 | train 26.9129 | val 27.4626 | lr 1.00e-06
Time (RNN): 38.84s


================ LSTM TRAINING ================

Epoch 1 | train 372.5669 | val 217.6269 | lr 1.00e-03
Epoch 20 | train 26.9303 | val 27.4830 | lr 5.00e-04
Epoch 40 | train 26.9187 | val 27.4714 | lr 1.25e-04
Epoch 60 | train 26.9143 | val 27.4652 | lr 3.13e-05
Epoch 80 | train 26.9134 | val 27.4631 | lr 7.81e-06
Epoch 100 | train 26.9132 | val 27.4626 | lr 1.95e-06
Epoch 120 | train 26.9126 | val 27.4625 | lr 1.00e-06
Epoch 140 | train 26.9141 | val 27.4624 | lr 1.00e-06
Epoch 160 | train 26.9130 | val 27.4623 | lr 1.00e-06
Epoch 180 | train 26.9125 | val 27.4623 | lr 1.00e-06
Epoch 200 | train 26.9133 | val 27.4622 | lr 1.00e-06
Epoch 220 | train 26.9131 | val 27.4622 | lr 1.00e-06
Epoch 240 | train 26.9124 | val 27.4621 | lr 1.00e-06
Epoch 260 | train 26.9126 | val 27.4621 | lr 1.00e-06
Epoch 280 | train 26.9133 | val 27.4621 | lr 1.00e-06
Epoch 300 | train 26.9134 | val 27.4621 | lr 1.00e-06
Epoch 320 | train 26.9132 | val 27.4620 | lr 1.00e-06
Epoch 340 | train 26.9131 | val 27.4620 | lr 1.00e-06
Epoch 360 | train 26.9129 | val 27.4620 | lr 1.00e-06
Epoch 380 | train 26.9132 | val 27.4620 | lr 1.00e-06
Epoch 400 | train 26.9132 | val 27.4620 | lr 1.00e-06
Time (LSTM): 78.16s


================ PREDIKCIJE SLEDEĆE LOTO KOMBINACIJE ================

Predikcija (MLP): [5, 10, 15, 20, x, y, z]
Predikcija (RNN): [5, 10, 15, 20, x, y, z]
Predikcija (LSTM): [5, 10, 15, 20, x, y, z]

✅ ALL EXPERIMENTS COMPLETED
============================================================
"""



"""
Predikcija: umesto dataset[-1][0] (što je predposlednji red), koristi se get_input_for_next_prediction() = poslednji red u fajlu
Vremenski podela: 85% train / 15% val, shuffle=False
Do max 400 epoha sa early stopping (patience 35), ReduceLROnPlateau, log svakih 20 epoha
Jači MLP (256→128 + dropout), RNN/LSTM sa hidden 128, 2 sloja, mali dropout između slojeva
num_workers=0 (manje problema na macOS), batch 48

"""

