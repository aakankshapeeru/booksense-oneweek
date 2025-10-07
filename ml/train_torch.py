import os, joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocess import load_dataset, build_preprocessor, make_xy, label_encoder_fit

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
DATA_PATH = os.path.join(ROOT, "data", "books.csv")

MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "torch_model.pt")
AUX_PATH   = os.path.join(MODEL_DIR, "torch_aux.joblib")

class NPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class TorchMLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    def forward(self, x): return self.net(x)

def safe_split(X, y):
    """Ensure the test set is valid for stratification on tiny datasets."""
    n = len(y)
    classes = np.unique(y)
    k = len(classes)

    # If any class has <2 samples, stratified split will fail—drop stratify.
    counts = {c: int(np.sum(y == c)) for c in classes}
    if any(v < 2 for v in counts.values()):
        return train_test_split(X, y, test_size=max(0.2, min(0.4, 1.0/n)), random_state=42, stratify=None)

    # With stratify, test_size must be >= num classes
    # Compute a float that guarantees ceil(test_size*n) >= k
    min_test_frac = max(0.2, k / n + 1e-9)
    try:
        return train_test_split(X, y, test_size=min_test_frac, random_state=42, stratify=y)
    except ValueError:
        # Final fallback: no stratify
        return train_test_split(X, y, test_size=min_test_frac, random_state=42, stratify=None)

def main():
    # Load & preprocess
    df = load_dataset(DATA_PATH)
    X_df, y = make_xy(df)
    y_idx, to_idx, to_lbl = label_encoder_fit(y)

    pre = build_preprocessor(df)
    X_enc = pre.fit_transform(X_df)
    X_enc = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc

    # Train/val split (robust to tiny data)
    X_train, X_val, y_train, y_val = safe_split(X_enc, y_idx.values)

    train_ds = NPDataset(np.array(X_train), np.array(y_train))
    val_ds   = NPDataset(np.array(X_val),   np.array(y_val))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)

    n_features = X_enc.shape[1]
    n_classes  = len(to_idx)

    model = TorchMLP(n_features, n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def evaluate(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return (correct / total) if total else 0.0

    epochs = 15
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        acc = evaluate(val_loader)
        print(f"Epoch {ep:02d}/{epochs} | val_acc={acc:.3f}")

    # Save weights + aux (preprocessor, label map, dims)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(
        {"preprocessor": pre, "label_map": to_lbl, "n_features": n_features, "n_classes": n_classes},
        AUX_PATH
    )
    print(f"Saved torch model weights to {MODEL_PATH} and aux to {AUX_PATH}")

if __name__ == "__main__":
    main()
