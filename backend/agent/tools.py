from __future__ import annotations

from typing import TypeVar, overload

import requests
from api.schemas import CreditAssessment, ExpenseCategorization, TaxCalculationResult
from langchain_core.tools import tool
from logger import GLOBAL_LOGGER as log
from pydantic import BaseModel
from rag import extract_text, get_llm

_T = TypeVar("_T", bound=BaseModel)


@overload
def _ask_llm(prompt: str) -> str: ...
@overload
def _ask_llm[T: BaseModel](prompt: str, response_format: type[_T]) -> _T: ...
def _ask_llm[T: BaseModel](prompt: str, response_format: type[_T] | None = None) -> _T | str:
    try:
        llm = get_llm()

        if response_format is not None:
            log.debug(
                "Using structured output model for LLM response",
                response_format=getattr(response_format, "__name__", str(response_format)),
            )
            structured_llm = llm.with_structured_output(response_format)
            return structured_llm.invoke(prompt)  # type: ignore[return-value]

        response = llm.invoke(prompt)
        return extract_text(response.content)

    except Exception as e:
        log.error("Error invoking LLM", error=str(e))
        raise RuntimeError(f"Error invoking LLM: {e!s}") from e


# tools

@tool
def check_sanctions_list(vendor_name: str) -> str:
    """Check if a vendor is on global restricted/sanctioned entities lists."""
    prompt = f"""You are a compliance screening system. Your job is to check whether the 
vendor name below appears on any major global sanctions or restricted-entity lists based 
on your training knowledge.

Check against these lists:
- US OFAC SDN (Specially Designated Nationals) List
- EU Consolidated Sanctions List
- UN Security Council Sanctions List
- UK HMT Sanctions List
- Any other major international sanctions or embargo lists

Vendor to screen: "{vendor_name}"

INSTRUCTIONS:
- If the vendor name matches or closely resembles ANY known sanctioned entity, respond 
  with: RED ALERT: [reason and which list they appear on]
- If the vendor is NOT found on any known sanctions list, respond with: 
  CLEARED: Vendor not found on known global sanctions lists.
- If the name is suspicious or could be a shell company for a sanctioned entity, note 
  that concern.
- Be precise. Do not fabricate matches — only flag if you have genuine knowledge.

Respond in a single concise paragraph starting with either "RED ALERT:" or "CLEARED:"."""

    try:
        return _ask_llm(prompt)
    except Exception as e:
        return f"SYSTEM WARNING: Sanctions check unavailable ({e}). Manual review required."


@tool
def get_vendor_credit_score(vendor_name: str) -> str:
    """Get the vendor creditworthiness assessment based on the publicly known financial health and credit history of the vendor."""
    vendor_name = vendor_name.strip()[:200]
    prompt = f"""You are a financial credit assessment system. Provide a credit score 
estimate (0-100) for the vendor below based on your training knowledge.

Vendor: "{vendor_name}"

ASSESSMENT CRITERIA:
- Company size, age, and market presence
- Industry reputation and stability
- Any known financial difficulties, bankruptcies, or controversies
- Public financial data or credit ratings if known (e.g., Dun & Bradstreet, S&P)
- Country of incorporation and regulatory environment

SCORING GUIDE:
- 75-100: Well-known, financially stable company → LOW risk
- 50-74: Mid-tier or limited public data → MODERATE risk  
- 0-49: Unknown, controversial, or financially distressed → HIGH risk

If the vendor is completely unknown, assign a score of 45 (MODERATE risk) and note that insufficient data is available."""

    try:
        assessment: CreditAssessment = _ask_llm(prompt, response_format=CreditAssessment)
        return (
            f"CREDIT SCORE: {assessment.score}/100 | "
            f"RISK: {assessment.risk_level.upper()} | "
            f"REASONING: {assessment.reasoning}"
        )
    except Exception as e:
        return f"SYSTEM WARNING: Credit check unavailable ({e}). Manual review required."


## Agent 2 Tools: Tax & Treasury ---

_TAX_FALLBACK_RATE = 0.15

@tool
def calculate_cross_border_tax(amount: float, origin: str, destination: str) -> dict:
    """Calculates VAT/GST and import duties for cross-border transactions.
    Returns a dict with keys: route, rate, tax_amount, total_cost, trade_agreement_notes.
    On LLM or parsing failure, falls back to a flat 15% estimate and notes the reason.
    """

    if amount < 0:
        raise ValueError(f"Transaction amount must be non-negative, got {amount}")

    origin = origin.strip().upper()[:10]
    destination = destination.strip().upper()[:10]

    if not origin or not destination:
        raise ValueError("Both origin and destination country codes must be provided.")

    prompt = f"""You are an international tax calculation system. Calculate the applicable 
taxes for the following cross-border transaction:

- Transaction Amount (EUR): {amount:,.2f}
- Origin Country Code: {origin}
- Destination Country Code: {destination}

CALCULATE:
1. The standard VAT/GST rate applicable in the destination country
2. Any applicable import duties for the product category (use general merchandise rate)
3. Consider any bilateral trade agreements between origin and destination
4. The combined effective tax rate

Key German/EU rates to apply:
- Standard German VAT: 19% for most goods
- Intra-EU B2B supply with valid VAT-ID: 0% (reverse charge)
- Import from outside EU: German import VAT 19% + applicable EU customs duty

All monetary amounts must be in EUR. All rates must be decimals between 0 and 1 (e.g. 0.19 for 19%).

Respond ONLY with the JSON object matching the schema — no other text."""

    try:
        tax_result: TaxCalculationResult = _ask_llm(prompt, response_format=TaxCalculationResult)

        return {
            "route": tax_result.route or f"{origin}_{destination}",
            "rate": tax_result.tax_rate,
            "vat_gst_rate": tax_result.vat_gst_rate,
            "import_duty_rate": tax_result.import_duty_rate,
            "tax_amount": tax_result.tax_amount,
            "vat_gst_amount": tax_result.vat_gst_amount,
            "import_duty_amount": tax_result.import_duty_amount,
            "total_cost": tax_result.total_cost,
            "trade_agreement_notes": tax_result.trade_agreement_notes,
        }

    except Exception as e:
        # LLM returned a structurally invalid response — log and use safe fallback
        log.warning(
            "Tax calculation: LLM response failed schema validation, using fallback rate",
            origin=origin,
            destination=destination,
            amount=amount,
            error=str(e),
        )
        tax_amount = round(amount * _TAX_FALLBACK_RATE, 2)
        return {
            "route": f"{origin}_{destination}",
            "rate": _TAX_FALLBACK_RATE,
            "vat_gst_rate": _TAX_FALLBACK_RATE,
            "import_duty_rate": 0.0,
            "tax_amount": tax_amount,
            "vat_gst_amount": tax_amount,
            "import_duty_amount": 0.0,
            "total_cost": round(amount + tax_amount, 2),
            "trade_agreement_notes": "Fallback flat rate applied — LLM response could not be validated. Manual review required.",
        }


@tool
def validate_fx_hedge(currency_pair: str, rate_used: float) -> str:
    """Checks if the FX rate used matches the current market rate, with a 5% variance threshold."""

    base, quote = None, None
    # Parse currency pair like "EUR_INR" or "USD_INR"
    parts = currency_pair.upper().replace("/", "_").replace("-", "_").split("_")
    if len(parts) == 2:
        base, quote = parts[0], parts[1]

    market_rate = None
    source = "live market"

    # --- Attempt 1: Live rate from Frankfurter API ---
    if base and quote:
        try:
            url = f"https://api.frankfurter.dev/v1/latest?base={base}&symbols={quote}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                market_rate = data.get("rates", {}).get(quote)
        except Exception:
            pass  # Fall through to LLM fallback

    # --- Attempt 2: LLM fallback for rate estimation ---
    if market_rate is None:
        prompt = f"""What is the approximate current exchange rate for {currency_pair.replace('_', '/')}?
Respond with ONLY a single number (the rate). For example, if 1 EUR = 91.5 INR, respond: 91.5"""
        llm_result = _ask_llm(prompt)
        source = "LLM-estimated"  # the live API failed, so say so in the result
        try:
            market_rate = float(llm_result.strip())
        except ValueError:
            return f"FX WARNING: Unable to determine market rate for '{currency_pair}'. Manual Treasury review required. (API and LLM both unavailable)"

    # --- Calculate variance ---
    variance = abs(rate_used - market_rate) / market_rate

    if variance > 0.05:
        return (
            f"FX ALERT: Rate used ({rate_used}) vs {source} rate ({market_rate:.4f}) — "
            f"Variance {variance:.2%} EXCEEDS 5% limit. Audit required before payment release."
        )
    return (
        f"FX SUCCESS: Rate used ({rate_used}) vs {source} rate ({market_rate:.4f}) — "
        f"Variance {variance:.2%} is within acceptable 5% hedge band."
    )

# --- Agent 3 Tools: Financial Control ---

@tool
def categorize_expense(amount: float, item_description: str) -> str:
    """Determines if the expense is Capital Expenditure (CapEx) or Operational (OpEx) using accounting standards."""
    if amount < 0:
        raise ValueError(f"Expense amount must be non-negative, got {amount}")

    item_description = item_description.strip()[:500]
    if not item_description:
        raise ValueError("Item description must not be empty.")

    prompt = f"""You are a financial controller applying GAAP/IFRS accounting standards.
Classify the following expense:

- Amount: {amount:,.2f} EUR
- Item Description: "{item_description}"

CLASSIFICATION RULES (GAAP/IFRS):
- CapEx: Assets with useful life > 1 year, that provide future economic benefit.
  Examples: servers, machinery, buildings, vehicles, large software licenses.
  → Must be capitalized on the balance sheet and depreciated.
  
- OpEx: Day-to-day operational costs consumed within the period.
  Examples: office supplies, SaaS subscriptions, repairs, utilities, travel.
  → Deducted immediately in the P&L.

ADDITIONAL CHECKS:
- If CapEx and amount > 500,000: Flag for additional CFO sign-off
- If classified as OpEx but amount > 100,000: Flag as possible misclassification
- Suggest depreciation period if CapEx (standard periods: IT equipment 3-5 years, 
  machinery 5-10 years, buildings 20-40 years)

Respond ONLY with the structured output matching the schema."""

    try:
        cat_result: ExpenseCategorization = _ask_llm(prompt, response_format=ExpenseCategorization)
        return (
            f"CLASSIFICATION: {cat_result.classification}\n"
            f"REASONING: {cat_result.reasoning}\n"
            f"DEPRECIATION: {cat_result.depreciation}\n"
            f"FLAGS: {cat_result.flags}"
        )
    except Exception as e:
        log.warning(
            "Expense categorization: LLM call or schema validation failed, using heuristic fallback",
            amount=amount,
            item_description=item_description,
            error=str(e),
        )
        if amount > 50000 or any(
            kw in item_description.lower()
            for kw in ["server", "machine", "vehicle", "building", "equipment", "hardware", "line", "tooling"]
        ):
            return (
                "CLASSIFICATION: CapEx\n"
                "REASONING: High value / capital equipment keyword detected (heuristic fallback)\n"
                "DEPRECIATION: 3-5 years\n"
                f"FLAGS: SYSTEM WARNING: Heuristic fallback used — LLM unavailable ({e})"
            )
        return (
            "CLASSIFICATION: OpEx\n"
            "REASONING: Consumable / operational expense inferred (heuristic fallback)\n"
            "DEPRECIATION: N/A\n"
            f"FLAGS: SYSTEM WARNING: Heuristic fallback used — LLM unavailable ({e})"
        )