"""
SQL-style / Pandas preprocessing pipeline.
Produces train/val/test splits as PyTorch-ready tensors.
"""
import re
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(__file__).parent
PROJECT_ROOT = DATA_DIR.parent

MAX_VOCAB = 2000
MAX_SEQ_LEN = 64

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def build_vocab(texts, max_size=MAX_VOCAB):
    word_counts = {}
    for t in texts:
        for w in tokenize(t):
            word_counts[w] = word_counts.get(w, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in sorted_words[:max_size - 2]:
        vocab[w] = len(vocab)
    return vocab

def text_to_indices(text, vocab, max_len=MAX_SEQ_LEN):
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens][:max_len]
    if len(indices) < max_len:
        indices = indices + [vocab["<pad>"]] * (max_len - len(indices))
    return indices

def preprocess():
    df = pd.read_csv(DATA_DIR / "claims.csv")

    # ---- Structured features ----
    # Numeric
    numeric_cols = ["patient_age", "length_of_stay", "claim_amount", "num_diagnoses", "num_procedures"]
    # Derive features
    df["cost_per_day"] = df["claim_amount"] / df["length_of_stay"].clip(lower=1)
    df["has_secondary_dx"] = (df["secondary_diagnoses"] != "").astype(int)
    numeric_cols += ["cost_per_day", "has_secondary_dx"]

    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(df[numeric_cols])

    # Categorical buckets
    df["primary_dx_group"] = df["primary_diagnosis"].str[0:1]  # e.g., I, J, E
    df["high_cost_flag"] = (df["claim_amount"] > df["claim_amount"].quantile(0.8)).astype(int)
    df["long_stay_flag"] = (df["length_of_stay"] > 5).astype(int)

    cat_cols = ["discharge_disposition", "primary_dx_group"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_data = encoder.fit_transform(df[cat_cols])

    binary_cols = ["high_cost_flag", "long_stay_flag", "has_secondary_dx"]
    binary_data = df[binary_cols].values.astype(np.float32)

    structured = np.concatenate([numeric_data, cat_data, binary_data], axis=1).astype(np.float32)

    # ---- Text ----
    vocab = build_vocab(df["note_text"].tolist())
    text_indices = np.array([text_to_indices(t, vocab) for t in df["note_text"].tolist()], dtype=np.int64)

    # ---- Targets ----
    targets = df[["claim_denied", "missed_billing_flag"]].values.astype(np.float32)
    raw_amounts = df["claim_amount"].values.astype(np.float32)

    # ---- Train/Val/Test split ----
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=SEED, stratify=targets[:, 0])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED, stratify=targets[temp_idx, 0])

    splits = {
        "train": (train_idx, structured[train_idx], text_indices[train_idx], targets[train_idx], raw_amounts[train_idx]),
        "val":   (val_idx,   structured[val_idx],   text_indices[val_idx],   targets[val_idx],   raw_amounts[val_idx]),
        "test":  (test_idx,  structured[test_idx],  text_indices[test_idx],  targets[test_idx],  raw_amounts[test_idx]),
    }

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (idxs, s, t, y, amt) in splits.items():
        torch.save({
            "indices": idxs,
            "structured": torch.from_numpy(s),
            "text": torch.from_numpy(t),
            "targets": torch.from_numpy(y),
            "claim_amounts": torch.from_numpy(amt),
        }, out_dir / f"{split_name}.pt")

    # Save artifacts
    pickle.dump(scaler, open(out_dir / "scaler.pkl", "wb"))
    pickle.dump(encoder, open(out_dir / "encoder.pkl", "wb"))
    pickle.dump(vocab, open(out_dir / "vocab.pkl", "wb"))
    pickle.dump({"numeric": numeric_cols, "cat": cat_cols, "binary": binary_cols}, open(out_dir / "feature_names.pkl", "wb"))

    print(f"Saved processed data -> {out_dir}")
    print(f"Structured dim: {structured.shape[1]}, Vocab size: {len(vocab)}")

if __name__ == "__main__":
    preprocess()
