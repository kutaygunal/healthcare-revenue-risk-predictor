from pydantic import BaseModel
from typing import List, Optional

class ClaimRequest(BaseModel):
    patient_age: int
    length_of_stay: int
    claim_amount: float
    num_diagnoses: int
    num_procedures: int
    primary_diagnosis: str
    secondary_diagnoses: Optional[str] = ""
    procedure_codes: Optional[str] = ""
    principal_procedure: Optional[str] = ""
    discharge_disposition: str
    note_text: str

class ExplanationItem(BaseModel):
    feature: str
    denial_impact: float
    missed_revenue_impact: float

class TextTokenItem(BaseModel):
    token: str
    attention_weight: float

class ClaimResponse(BaseModel):
    claim_denial_risk: float
    missed_revenue_risk: float
    denial_explanation: List[ExplanationItem]
    text_highlights: List[TextTokenItem]
    estimated_recoverable_revenue: float
    message: str
