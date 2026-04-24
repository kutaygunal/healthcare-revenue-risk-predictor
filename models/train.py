"""
Training loop for RevenueRiskNet.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from model import RevenueRiskNet, bce_with_logits

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5

def load_split(name):
    d = torch.load(DATA_DIR / f"{name}.pt", map_location="cpu")
    return d["structured"], d["text"], d["targets"], d["claim_amounts"]

def create_loaders():
    train_s, train_t, train_y, _ = load_split("train")
    val_s, val_t, val_y, _ = load_split("val")
    test_s, test_t, test_y, _ = load_split("test")

    train_ds = torch.utils.data.TensorDataset(train_s, train_t, train_y)
    val_ds = torch.utils.data.TensorDataset(val_s, val_t, val_y)
    test_ds = torch.utils.data.TensorDataset(test_s, test_t, test_y)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

def evaluate(model, loader):
    model.eval()
    all_denial_true, all_denial_pred = [], []
    all_missed_true, all_missed_pred = [], []
    total_loss = 0.0
    with torch.no_grad():
        for s, t, y in loader:
            s, t, y = s.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
            d_logit, m_logit, _ = model(s, t)
            loss = bce_with_logits(d_logit, y[:, 0:1]) + bce_with_logits(m_logit, y[:, 1:2])
            total_loss += loss.item()

            all_denial_true.append(y[:, 0].cpu().numpy())
            all_denial_pred.append(torch.sigmoid(d_logit).cpu().numpy().ravel())
            all_missed_true.append(y[:, 1].cpu().numpy())
            all_missed_pred.append(torch.sigmoid(m_logit).cpu().numpy().ravel())

    dt = np.concatenate(all_denial_true)
    dp = np.concatenate(all_denial_pred)
    mt = np.concatenate(all_missed_true)
    mp = np.concatenate(all_missed_pred)

    metrics = {
        "loss": total_loss / len(loader),
        "denial_auc": roc_auc_score(dt, dp),
        "denial_f1": f1_score(dt, dp > 0.5),
        "denial_acc": accuracy_score(dt, dp > 0.5),
        "missed_auc": roc_auc_score(mt, mp),
        "missed_f1": f1_score(mt, mp > 0.5),
        "missed_acc": accuracy_score(mt, mp > 0.5),
    }
    return metrics

def train():
    train_loader, val_loader, test_loader = create_loaders()

    # Determine dimensions
    sample_s, _, _, _ = load_split("train")
    vocab = pickle.load(open(DATA_DIR / "vocab.pkl", "rb"))
    struct_dim = sample_s.shape[1]
    vocab_size = len(vocab)

    model = RevenueRiskNet(struct_dim, vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for s, t, y in train_loader:
            s, t, y = s.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            d_logit, m_logit, _ = model(s, t)
            loss = bce_with_logits(d_logit, y[:, 0:1]) + bce_with_logits(m_logit, y[:, 1:2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics = evaluate(model, val_loader)
        val_auc = (val_metrics["denial_auc"] + val_metrics["missed_auc"]) / 2
        scheduler.step(val_auc)

        print(f"Epoch {epoch:02d} | train_loss={epoch_loss/len(train_loader):.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"denial_auc={val_metrics['denial_auc']:.3f} missed_auc={val_metrics['missed_auc']:.3f} "
              f"denial_f1={val_metrics['denial_f1']:.3f} missed_f1={val_metrics['missed_f1']:.3f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "revenue_risk_model.pt")
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\n--- Test Evaluation ---")
    model.load_state_dict(torch.load(MODEL_DIR / "revenue_risk_model.pt", map_location=DEVICE))
    test_metrics = evaluate(model, test_loader)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    train()
