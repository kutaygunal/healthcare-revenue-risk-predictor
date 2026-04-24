"""
FastAPI inference endpoint for Healthcare Revenue Risk Predictor.
POST /predict-denial-risk
"""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI

from api.schemas import ClaimRequest, ClaimResponse, ExplanationItem, TextTokenItem
from models.model import RevenueRiskNet
from utils.explainability import SimpleExplainer

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "revenue_risk_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Healthcare Revenue Risk Predictor")

# ---- Load artifacts ----
vocab = pickle.load(open(DATA_DIR / "vocab.pkl", "rb"))
scaler = pickle.load(open(DATA_DIR / "scaler.pkl", "rb"))
encoder = pickle.load(open(DATA_DIR / "encoder.pkl", "rb"))
feature_meta = pickle.load(open(DATA_DIR / "feature_names.pkl", "rb"))

sample_structured = torch.load(DATA_DIR / "train.pt")["structured"]
STRUCT_DIM = sample_structured.shape[1]

model = RevenueRiskNet(STRUCT_DIM, len(vocab)).to(DEVICE)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("WARNING: Model checkpoint not found. Using randomly initialized model.")
model.eval()

explainer = SimpleExplainer(model)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def text_to_indices(text, max_len=64):
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens][:max_len]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    return indices

def preprocess_request(req: ClaimRequest):
    """Turn API payload into model-ready tensors."""
    # Build DataFrame row for sklearn transforms
    row = pd.DataFrame([{
        "patient_age": req.patient_age,
        "length_of_stay": req.length_of_stay,
        "claim_amount": req.claim_amount,
        "num_diagnoses": req.num_diagnoses,
        "num_procedures": req.num_procedures,
        "cost_per_day": req.claim_amount / max(req.length_of_stay, 1),
        "has_secondary_dx": 1 if req.secondary_diagnoses and req.secondary_diagnoses.strip() else 0,
        "primary_dx_group": req.primary_diagnosis[:1] if req.primary_diagnosis else "I",
        "high_cost_flag": 1 if req.claim_amount > 50000 else 0,
        "long_stay_flag": 1 if req.length_of_stay > 5 else 0,
        "discharge_disposition": req.discharge_disposition,
    }])

    numeric_cols = feature_meta["numeric"]
    cat_cols = feature_meta["cat"]
    binary_cols = feature_meta["binary"]

    numeric = scaler.transform(row[numeric_cols])
    cat = encoder.transform(row[cat_cols])
    binary = row[binary_cols].values.astype(np.float32)

    structured = np.concatenate([numeric, cat, binary], axis=1).astype(np.float32)
    structured_tensor = torch.from_numpy(structured).to(DEVICE)

    text_tensor = torch.tensor([text_to_indices(req.note_text)], dtype=torch.long).to(DEVICE)
    return structured_tensor, text_tensor

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict-denial-risk", response_model=ClaimResponse)
def predict_denial_risk(req: ClaimRequest):
    structured, text = preprocess_request(req)

    with torch.no_grad():
        denial_logit, missed_logit, text_attn = model(structured, text)

    denial_prob = float(torch.sigmoid(denial_logit).item())
    missed_prob = float(torch.sigmoid(missed_logit).item())

    # Explainability
    explanation = explainer.explain(structured, text, denial_logit, missed_logit, text_attn)

    # Business KPI: estimated recoverable revenue
    recovery_rate = 0.6
    estimated_recoverable = req.claim_amount * missed_prob * recovery_rate

    return ClaimResponse(
        claim_denial_risk=round(denial_prob, 4),
        missed_revenue_risk=round(missed_prob, 4),
        denial_explanation=[ExplanationItem(**item) for item in explanation["top_structured_features"]],
        text_highlights=[TextTokenItem(token=t[0], attention_weight=round(t[1], 4)) for t in explanation["top_text_tokens"]],
        estimated_recoverable_revenue=round(estimated_recoverable, 2),
        message="Prediction complete. Review high-attention clinical terms and top structured risk drivers."
    )
