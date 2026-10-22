from typing import Literal

from pydantic import BaseModel, Field

# ============================================================
# AGENT  Schemas
# ============================================================


class AuditRequest(BaseModel):
    request_text: str


class AuditResponse(BaseModel):
    risk_result: str
    tax_result: str
    control_result: str
    cfo_memo: str


# ============================================================
# RAG  Schemas
# ============================================================


class QueryRequest(BaseModel):
    query: str
    retriever_type: str = "similarity"


class QueryResponse(BaseModel):
    answer: str


# ============================================================
# Vendor Credit Score Schemas
# ============================================================
class CreditAssessment(BaseModel):
    score: int = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MODERATE", "HIGH"]
    reasoning: str

class TaxCalculationResult(BaseModel):
    tax_rate: float = Field(ge=0, le=1)
    vat_gst_rate: float = Field(default=0.0, ge=0, le=1)
    import_duty_rate: float = Field(default=0.0, ge=0, le=1)
    tax_amount: float = Field(description="Calculated as amount * tax_rate")
    vat_gst_amount: float = Field(default=0.0)
    import_duty_amount: float = Field(default=0.0)
    total_cost: float = Field(description="Calculated as original amount + total tax")
    trade_agreement_notes: str = Field(default="", description="Any relevant trade agreement information")
    route: str = Field(default="", description="Format: {origin}_{destination}")


# ============================================================
# Expense Categorization Schemas
# ============================================================
class ExpenseCategorization(BaseModel):
    classification: Literal["CapEx", "OpEx"] = Field(
        description="Classification of the expense: CapEx (capital expenditure) or OpEx (operational expense)"
    )
    reasoning: str = Field(
        description="Accounting rationale under GAAP/IFRS explaining why this classification applies"
    )
    depreciation: str = Field(
        default="N/A",
        description="Depreciation period if CapEx (e.g., '3-5 years', '5-10 years'), or 'N/A' if OpEx"
    )
    flags: str = Field(
        default="None",
        description="Anomaly flags or sign-off requirements (e.g. CFO sign-off if CapEx > 500k, OpEx > 100k anomaly), or 'None'"
    )