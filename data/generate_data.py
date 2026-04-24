"""
Generate synthetic healthcare claims data + clinical discharge notes.
Mimics hospital revenue-cycle structured + text data.
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N = 2000
OUTPUT_DIR = Path(__file__).parent

# ICD-10 categories (simplified)
ICD_POOL = [f"I{i:02d}" for i in range(1, 26)] + [f"J{i:02d}" for i in range(1, 21)] + [f"E{i:02d}" for i in range(10, 16)]
CPT_POOL = [str(10000 + i) for i in range(0, 500, 10)]

NOTE_TEMPLATES = [
    "Patient admitted for {condition}. {procedure} performed. Length of stay {los} days. Discharged in stable condition with {complication}.",
    "{age}-year-old patient presented with {condition}. Underwent {procedure}. Course complicated by {complication}. Discharged after {los} days.",
    "Admission for {condition}. {procedure} completed without issues. {los} day stay. Discharge instructions given. {complication} noted.",
    "Patient with history of {condition} required {procedure}. {complication} observed during stay. Discharged on day {los}.",
    "Elective {procedure} for {condition}. Uneventful recovery. LOS {los}. Discharge: {complication}.",
]

CONDITIONS = [
    "acute myocardial infarction", "pneumonia", "congestive heart failure", "sepsis",
    "chronic obstructive pulmonary disease", "hip fracture", "stroke", "diabetic ketoacidosis",
    "gastrointestinal bleeding", "cellulitis"
]
PROCEDURES = [
    "percutaneous coronary intervention", "mechanical ventilation", "hip replacement",
    "diagnostic cardiac catheterization", "colonoscopy", "appendectomy", "bronchoscopy",
    "lumbar puncture", "wound debridement", "central line placement"
]
COMPLICATIONS = [
    "no complications", "acute kidney injury", "respiratory failure",
    "surgical site infection", "delayed wound healing", "no complications",
    "mild electrolyte imbalance", "transient arrhythmia", "no complications"
]

def generate_note(age, los, condition, procedure, complication):
    template = random.choice(NOTE_TEMPLATES)
    return template.format(
        age=int(age), los=int(los), condition=condition,
        procedure=procedure, complication=complication
    )

def assign_risk_scores(row):
    """Deterministic logic to create plausible denial / missed-revenue labels."""
    denial_score = 0.0
    missed_score = 0.0

    # Denial risk factors
    if row["claim_amount"] > 50000:
        denial_score += 0.25
    if row["length_of_stay"] > 7:
        denial_score += 0.20
    if row["num_procedures"] == 0 and row["claim_amount"] > 10000:
        denial_score += 0.30
    if row["discharge_disposition"] == "AMA":
        denial_score += 0.35
    if "sepsis" in row["primary_diagnosis_text"] or "AMI" in row["primary_diagnosis"]:
        denial_score += 0.10
    if row["claim_amount"] / max(row["length_of_stay"], 1) > 8000:
        denial_score += 0.15

    # Missed revenue risk factors
    if row["num_diagnoses"] <= 1 and row["length_of_stay"] > 3:
        missed_score += 0.30
    if row["num_procedures"] < 2 and row["claim_amount"] < 5000 and row["length_of_stay"] > 2:
        missed_score += 0.20
    if row["principal_procedure"] == "NONE":
        missed_score += 0.25
    if row["discharge_disposition"] == "Home" and row["length_of_stay"] > 5:
        missed_score += 0.10
    if "no complications" not in row["note_text"] and row["num_diagnoses"] < 2:
        missed_score += 0.15

    denial_prob = min(max(denial_score + np.random.normal(0, 0.08), 0.0), 1.0)
    missed_prob = min(max(missed_score + np.random.normal(0, 0.08), 0.0), 1.0)
    return denial_prob, missed_prob

def main():
    records = []
    for i in range(N):
        age = np.random.randint(18, 90)
        los = max(1, int(np.random.exponential(3)) + 1)
        claim_amount = max(1000, int(np.random.lognormal(mean=9, sigma=0.8)))
        num_diagnoses = np.random.randint(1, 6)
        num_procedures = np.random.randint(0, 4)
        primary_icd = random.choice(ICD_POOL)
        secondary_icds = random.choices(ICD_POOL, k=num_diagnoses - 1)
        cpts = random.choices(CPT_POOL, k=num_procedures)
        principal_proc = cpts[0] if cpts else "NONE"
        disposition = random.choices(["Home", "SNF", "Home Health", "AMA", "Expired"], weights=[0.6, 0.15, 0.15, 0.05, 0.05])[0]
        condition = random.choice(CONDITIONS)
        procedure = random.choice(PROCEDURES) if num_procedures > 0 else "observation"
        complication = random.choice(COMPLICATIONS)
        note = generate_note(age, los, condition, procedure, complication)

        record = {
            "claim_id": f"CLM_{i:06d}",
            "patient_age": age,
            "length_of_stay": los,
            "claim_amount": claim_amount,
            "num_diagnoses": num_diagnoses,
            "num_procedures": num_procedures,
            "primary_diagnosis": primary_icd,
            "secondary_diagnoses": ";".join(secondary_icds),
            "procedure_codes": ";".join(cpts),
            "principal_procedure": principal_proc,
            "discharge_disposition": disposition,
            "primary_diagnosis_text": condition,
            "note_text": note,
        }
        records.append(record)

    df = pd.DataFrame(records)
    df[["denial_risk", "missed_revenue_risk"]] = df.apply(assign_risk_scores, axis=1, result_type="expand")
    df["claim_denied"] = (df["denial_risk"] > 0.5).astype(int)
    df["missed_billing_flag"] = (df["missed_revenue_risk"] > 0.5).astype(int)

    df.to_csv(OUTPUT_DIR / "claims.csv", index=False)
    print(f"Generated {len(df)} synthetic claims -> {OUTPUT_DIR / 'claims.csv'}")

if __name__ == "__main__":
    main()
