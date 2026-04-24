"""
Streamlit business KPI dashboard for Healthcare Revenue Risk Predictor.
Simulates a Snowflake-style data warehouse view + model predictions.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import RevenueRiskNet
from api.main import DEVICE, vocab, scaler, encoder, feature_meta, text_to_indices

st.set_page_config(page_title="Revenue Risk KPI Dashboard", layout="wide")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "revenue_risk_model.pt"

@st.cache_resource
def load_model():
    sample_structured = torch.load(DATA_DIR / "train.pt")["structured"]
    struct_dim = sample_structured.shape[1]
    model = RevenueRiskNet(struct_dim, len(vocab)).to(DEVICE)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def predict_batch(df: pd.DataFrame):
    model = load_model()
    # Preprocess aligned with training pipeline
    df["cost_per_day"] = df["claim_amount"] / df["length_of_stay"].clip(lower=1)
    df["has_secondary_dx"] = (df["secondary_diagnoses"].fillna("") != "").astype(int)
    df["primary_dx_group"] = df["primary_diagnosis"].str[:1].fillna("I")
    df["high_cost_flag"] = (df["claim_amount"] > 50000).astype(int)
    df["long_stay_flag"] = (df["length_of_stay"] > 5).astype(int)

    numeric_cols = feature_meta["numeric"]
    cat_cols = feature_meta["cat"]
    binary_cols = feature_meta["binary"]

    numeric = scaler.transform(df[numeric_cols])
    cat = encoder.transform(df[cat_cols])
    binary = df[binary_cols].values.astype(np.float32)
    structured = torch.from_numpy(np.concatenate([numeric, cat, binary], axis=1).astype(np.float32)).to(DEVICE)

    texts = df["note_text"].fillna("").tolist()
    text_indices = torch.tensor([text_to_indices(t) for t in texts], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        d_logit, m_logit, _ = model(structured, text_indices)
    denial_probs = torch.sigmoid(d_logit).cpu().numpy().ravel()
    missed_probs = torch.sigmoid(m_logit).cpu().numpy().ravel()
    return denial_probs, missed_probs

def main():
    st.title("Healthcare Revenue Risk KPI Dashboard")
    st.markdown("Simulated revenue-cycle analytics. Upload claims CSV or use synthetic data.")

    data_source = st.radio("Data source", ["Synthetic Demo", "Upload CSV"])

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload claims CSV", type="csv")
        if uploaded is None:
            st.info("Upload a CSV with the expected columns.")
            return
        df = pd.read_csv(uploaded)
    else:
        claims_path = PROJECT_ROOT / "data" / "claims.csv"
        if not claims_path.exists():
            st.error("Synthetic data not found. Run `python data/generate_data.py` first.")
            return
        df = pd.read_csv(claims_path)

    required = ["patient_age", "length_of_stay", "claim_amount", "num_diagnoses",
                "num_procedures", "primary_diagnosis", "secondary_diagnoses",
                "procedure_codes", "principal_procedure", "discharge_disposition", "note_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    if st.button("Run Predictions"):
        with st.spinner("Scoring claims..."):
            denial_probs, missed_probs = predict_batch(df)
        df["denial_risk"] = denial_probs
        df["missed_revenue_risk"] = missed_probs

        recovery_rate = 0.6
        df["recoverable_revenue"] = df["claim_amount"] * df["missed_revenue_risk"] * recovery_rate

        # ---- KPIs ----
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Claims", f"{len(df):,}")
        col2.metric("Avg Denial Risk", f"{df['denial_risk'].mean():.1%}")
        col3.metric("Avg Missed Revenue Risk", f"{df['missed_revenue_risk'].mean():.1%}")
        col4.metric("Est. Recoverable Revenue", f"${df['recoverable_revenue'].sum():,.0f}")

        st.divider()

        # ---- High-risk tables ----
        high_denial = df[df["denial_risk"] > 0.7].sort_values("denial_risk", ascending=False).head(20)
        high_missed = df[df["missed_revenue_risk"] > 0.7].sort_values("missed_revenue_risk", ascending=False).head(20)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔴 Highest Denial Risk Claims")
            st.dataframe(high_denial[["claim_id", "patient_age", "claim_amount", "length_of_stay", "denial_risk"]].reset_index(drop=True))
        with c2:
            st.subheader("🟡 Highest Missed Revenue Risk")
            st.dataframe(high_missed[["claim_id", "patient_age", "claim_amount", "length_of_stay", "missed_revenue_risk", "recoverable_revenue"]].reset_index(drop=True))

        st.divider()

        # ---- Aggregates ----
        st.subheader("Risk by Discharge Disposition")
        agg = df.groupby("discharge_disposition").agg({
            "denial_risk": "mean",
            "missed_revenue_risk": "mean",
            "recoverable_revenue": "sum",
            "claim_id": "count"
        }).rename(columns={"claim_id": "count"}).sort_values("recoverable_revenue", ascending=False)
        st.bar_chart(agg[["denial_risk", "missed_revenue_risk"]])
        st.dataframe(agg.style.format({
            "denial_risk": "{:.1%}",
            "missed_revenue_risk": "{:.1%}",
            "recoverable_revenue": "${:,.0f}",
        }))

        # ---- Download ----
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Scored Claims", csv, "scored_claims.csv", "text/csv")

if __name__ == "__main__":
    main()
