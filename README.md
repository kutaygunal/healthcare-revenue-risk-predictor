# Healthcare Revenue Risk Predictor

A PyTorch-based healthcare revenue risk prediction system using **structured claim features** and **clinical text**. Built to mirror the type of clinical AI used in hospital revenue cycle management — like what SmarterDx builds.

## What It Does

| Input | Output |
|-------|--------|
| Patient age, diagnosis codes (ICD), procedure codes (CPT), length of stay, claim amount | Claim denial risk score |
| Discharge notes / clinical free-text | Missed revenue risk score |
| | Explanation of key predictive factors |

## Architecture

```
Healthcare Claims Data
    ↓
SQL / Pandas Preprocessing
    ↓
Structured Features + Clinical Text
    ↓
PyTorch Multi-Task Model
    ↓
Risk Scores + Explanation
    ↓
FastAPI Endpoint / Streamlit KPI Dashboard
```

## Tech Stack

- **Python**
- **PyTorch** — deep learning model (structured + text)
- **Pandas / SQL-style transforms** — feature engineering
- **FastAPI** — production-grade inference endpoint
- **Streamlit** — business KPI dashboard
- **scikit-learn** — preprocessing and metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python data/generate_data.py
```

Generates `data/claims.csv` and `data/clinical_notes.csv`.

### 3. Preprocess

```bash
python data/preprocess.py
```

Produces train/val/test tensors and vocab mappings.

### 4. Train the Model

```bash
python models/train.py
```

Saves `models/revenue_risk_model.pt` and metrics.

### 5. Run the API

```bash
uvicorn api.main:app --reload
```

POST to `http://localhost:8000/predict-denial-risk`

### 6. Launch the KPI Dashboard

```bash
streamlit run dashboard/kpi_dashboard.py
```

## Project Structure

```
.
├── data/
│   ├── generate_data.py      # Synthetic claims + notes
│   └── preprocess.py         # Feature engineering / SQL-style transforms
├── models/
│   ├── model.py              # PyTorch architecture
│   └── train.py              # Training loop
├── api/
│   ├── main.py               # FastAPI app
│   └── schemas.py            # Pydantic request/response models
├── dashboard/
│   └── kpi_dashboard.py      # Streamlit business dashboard
├── utils/
│   └── explainability.py     # SHAP-like explanations
├── requirements.txt
└── README.md
```

## Model Details

- **Structured encoder:** MLP over tabular features (age, LOS, claim amount, ICD/CPT buckets)
- **Text encoder:** Word Embedding → BiGRU → Attention → Context vector
- **Output heads:**
  - Claim Denial Risk (sigmoid)
  - Missed Revenue Risk (sigmoid)
- **Explainability:** Attention weights over clinical text + top-k structured feature contributions

## Business KPIs Tracked

- Average Denial Risk per Department
- Average Missed Revenue Risk
- Estimated Recoverable Revenue = Σ(claim_amount × missed_revenue_risk × recovery_rate)
- High-Risk Claim Count
