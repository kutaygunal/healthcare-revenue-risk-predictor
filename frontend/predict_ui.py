"""
Streamlit interactive UI for single-claim denial risk prediction.
Run: streamlit run frontend/predict_ui.py
"""
import math
import pickle
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import RevenueRiskNet
from utils.explainability import SimpleExplainer

st.set_page_config(page_title="Claim Risk Predictor", layout="wide")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "revenue_risk_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache model load
@st.cache_resource
def load_artifacts():
    vocab = pickle.load(open(DATA_DIR / "vocab.pkl", "rb"))
    scaler = pickle.load(open(DATA_DIR / "scaler.pkl", "rb"))
    encoder = pickle.load(open(DATA_DIR / "encoder.pkl", "rb"))
    feature_meta = pickle.load(open(DATA_DIR / "feature_names.pkl", "rb"))
    sample_structured = torch.load(DATA_DIR / "train.pt")["structured"]
    struct_dim = sample_structured.shape[1]

    model = RevenueRiskNet(struct_dim, len(vocab)).to(DEVICE)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    explainer = SimpleExplainer(model)
    return vocab, scaler, encoder, feature_meta, model, explainer

vocab, scaler, encoder, feature_meta, model, explainer = load_artifacts()

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def text_to_indices(text, max_len=64):
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens][:max_len]
    indices += [vocab["<pad>"]] * (max_len - len(indices))
    return indices

def preprocess_input(data):
    row = pd.DataFrame([{
        "patient_age": data["patient_age"],
        "length_of_stay": data["length_of_stay"],
        "claim_amount": data["claim_amount"],
        "num_diagnoses": data["num_diagnoses"],
        "num_procedures": data["num_procedures"],
        "cost_per_day": data["claim_amount"] / max(data["length_of_stay"], 1),
        "has_secondary_dx": 1 if data.get("secondary_diagnoses", "").strip() else 0,
        "primary_dx_group": data["primary_diagnosis"][:1] if data["primary_diagnosis"] else "I",
        "high_cost_flag": 1 if data["claim_amount"] > 50000 else 0,
        "long_stay_flag": 1 if data["length_of_stay"] > 5 else 0,
        "discharge_disposition": data["discharge_disposition"],
    }])

    numeric_cols = feature_meta["numeric"]
    cat_cols = feature_meta["cat"]
    binary_cols = feature_meta["binary"]

    numeric = scaler.transform(row[numeric_cols])
    cat = encoder.transform(row[cat_cols])
    binary = row[binary_cols].values.astype(np.float32)
    structured = torch.from_numpy(np.concatenate([numeric, cat, binary], axis=1).astype(np.float32)).to(DEVICE)
    text_tensor = torch.tensor([text_to_indices(data["note_text"])], dtype=torch.long).to(DEVICE)
    return structured, text_tensor

# ---- UI ----
st.title("Healthcare Revenue Risk Predictor")
st.markdown("Enter claim details below to predict denial risk, missed revenue risk, and key drivers.")

with st.form("claim_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=67)
        length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=60, value=5)
        claim_amount = st.number_input("Claim Amount ($)", min_value=0, value=85000, step=1000)
    with col2:
        num_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=1)
        num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=20, value=0)
        primary_diagnosis = st.text_input("Primary Diagnosis Code", value="I21")
    with col3:
        secondary_diagnoses = st.text_input("Secondary Diagnoses (semicolon separated)", value="")
        procedure_codes = st.text_input("Procedure Codes (semicolon separated)", value="")
        principal_procedure = st.text_input("Principal Procedure", value="NONE")

    discharge_disposition = st.selectbox(
        "Discharge Disposition",
        ["Home", "SNF", "Home Health", "AMA", "Expired"]
    )
    note_text = st.text_area(
        "Clinical / Discharge Note",
        value="67-year-old patient presented with acute myocardial infarction. Observation only. No procedure performed. 5 day stay. Discharge: no complications noted.",
        height=120
    )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    data = {
        "patient_age": int(patient_age),
        "length_of_stay": int(length_of_stay),
        "claim_amount": float(claim_amount),
        "num_diagnoses": int(num_diagnoses),
        "num_procedures": int(num_procedures),
        "primary_diagnosis": primary_diagnosis,
        "secondary_diagnoses": secondary_diagnoses,
        "procedure_codes": procedure_codes,
        "principal_procedure": principal_procedure,
        "discharge_disposition": discharge_disposition,
        "note_text": note_text,
    }

    structured, text = preprocess_input(data)

    with torch.no_grad():
        denial_logit, missed_logit, text_attn = model(structured, text)

    denial_prob = float(torch.sigmoid(denial_logit).item())
    missed_prob = float(torch.sigmoid(missed_logit).item())

    explanation = explainer.explain(structured, text, denial_logit, missed_logit, text_attn)

    recovery_rate = 0.6
    recoverable = claim_amount * missed_prob * recovery_rate

    st.divider()

    # Risk score cards
    c1, c2, c3 = st.columns(3)
    denial_color = "#ff4b4b" if denial_prob > 0.6 else "#ffa500" if denial_prob > 0.3 else "#00cc66"
    missed_color = "#ff4b4b" if missed_prob > 0.6 else "#ffa500" if missed_prob > 0.3 else "#00cc66"

    with c1:
        st.markdown(f"""
        <div style="padding:20px;border-radius:10px;background-color:{denial_color}20;border:2px solid {denial_color};">
            <h3 style="margin:0;color:{denial_color};">Claim Denial Risk</h3>
            <h1 style="margin:0;font-size:48px;color:{denial_color};">{denial_prob:.1%}</h1>
            <p style="margin:0;color:gray;">Probability insurance rejects claim</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="padding:20px;border-radius:10px;background-color:{missed_color}20;border:2px solid {missed_color};">
            <h3 style="margin:0;color:{missed_color};">Missed Revenue Risk</h3>
            <h1 style="margin:0;font-size:48px;color:{missed_color};">{missed_prob:.1%}</h1>
            <p style="margin:0;color:gray;">Under-coding / billing opportunity missed</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="padding:20px;border-radius:10px;background-color:#1f77b420;border:2px solid #1f77b4;">
            <h3 style="margin:0;color:#1f77b4;">Est. Recoverable Revenue</h3>
            <h1 style="margin:0;font-size:48px;color:#1f77b4;">${recoverable:,.0f}</h1>
            <p style="margin:0;color:gray;">60% recovery rate applied</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Explanations
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Structured Risk Drivers")
        for item in explanation["top_structured_features"]:
            d_imp = item["denial_impact"]
            m_imp = item["missed_revenue_impact"]
            d_dir = "↑" if d_imp > 0 else "↓" if d_imp < 0 else "→"
            m_dir = "↑" if m_imp > 0 else "↓" if m_imp < 0 else "→"
            st.markdown(f"""
            - **{item['feature']}**  
              Denial: {d_dir} `{d_imp:+.4f}` &nbsp;|&nbsp; Missed Revenue: {m_dir} `{m_imp:+.4f}`
            """)

    with col_b:
        st.subheader("High-Attention Clinical Terms")
        for token, weight in explanation["top_text_tokens"]:
            pct = min(weight * 300, 100)  # scale for bar
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin-bottom:4px;">
                <div style="width:120px;font-weight:500;">{token}</div>
                <div style="flex:1;background:#e0e0e0;border-radius:4px;height:20px;overflow:hidden;">
                    <div style="width:{pct:.0f}%;background:#1f77b4;height:100%;"></div>
                </div>
                <div style="width:60px;text-align:right;font-size:0.9em;">{weight:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.info("Tip: High denial risk? Double-check coding alignment between diagnoses and procedures. High missed revenue risk? Confirm all secondary diagnoses and complication codes were captured.")
