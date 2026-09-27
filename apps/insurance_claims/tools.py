from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BASE_DIR / "fixtures"


def _load_json(filename: str) -> Any:
    with open(FIXTURES_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_policyholders() -> List[Dict[str, Any]]:
    return _load_json("policyholders.json")


def load_claims() -> List[Dict[str, Any]]:
    return _load_json("claims.json")


def load_representatives() -> List[Dict[str, Any]]:
    return _load_json("representatives.json")


def load_document_guideline() -> Dict[str, Any]:
    return _load_json("required_document_guideline.json")


def load_claim_schema() -> Dict[str, Any]:
    return _load_json("claim_schema.json")


def load_consent_scenarios() -> Dict[str, Any]:
    return _load_json("consent_scenarios.json")


def _norm_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return " ".join(str(value).strip().lower().split())


def _norm_phone(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def _norm_email(value: Optional[str]) -> Optional[str]:
    return _norm_text(value)


def _norm_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%B %d %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%b %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value


def _value_matches(field: str, supplied: Optional[str], record: Dict[str, Any]) -> bool:
    if supplied in (None, ""):
        return False

    if field == "full_name":
        names = [record.get("name")] + record.get("name_aliases", [])
        supplied_n = _norm_text(supplied)
        return any(_norm_text(name) == supplied_n for name in names if name)

    if field == "dob":
        return _norm_date(supplied) == _norm_date(record.get("dob"))

    if field == "phone":
        phones = [record.get("phone")] + record.get("phone_aliases", [])
        supplied_n = _norm_phone(supplied)
        return any(_norm_phone(phone) == supplied_n for phone in phones if phone)

    if field == "email":
        emails = [record.get("email")] + record.get("email_aliases", [])
        supplied_n = _norm_email(supplied)
        return any(_norm_email(email) == supplied_n for email in emails if email)

    if field == "id_last4":
        return str(supplied).strip() == str(record.get("id_last4", "")).strip()

    return False


PII_FIELDS = ["full_name", "dob", "phone", "email", "id_last4"]


def find_policyholder_by_policy(policy_number: Optional[str]) -> Optional[Dict[str, Any]]:
    if not policy_number:
        return None
    target = _norm_text(policy_number)
    for person in load_policyholders():
        if _norm_text(person.get("policy_number")) == target:
            return person
    return None


def identify_policyholder(state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], int, List[str]]:
    """
    Returns (best_candidate, number_of_matching_PII_fields, matched_field_names).

    If policy_number is supplied, it narrows the search but does NOT count as PII.
    Without a policy number, the best unique PII match is chosen.
    """
    people = load_policyholders()
    if state.get("policy_number"):
        candidate = find_policyholder_by_policy(state.get("policy_number"))
        people = [candidate] if candidate else []

    scored: List[Tuple[Dict[str, Any], int, List[str]]] = []
    for person in people:
        if not person:
            continue
        matched = [
            field
            for field in PII_FIELDS
            if _value_matches(field, state.get(field), person)
        ]
        scored.append((person, len(matched), matched))

    if not scored:
        return None, 0, []

    scored.sort(key=lambda x: x[1], reverse=True)
    best_score = scored[0][1]
    best = [row for row in scored if row[1] == best_score]

    # Do not guess between equally-scored customers unless policy number already narrowed it.
    if len(best) > 1 and not state.get("policy_number"):
        return None, best_score, []

    return best[0]


def get_claims_for_party(party_id: str) -> List[Dict[str, Any]]:
    return [c for c in load_claims() if c.get("party_id") == party_id]


def get_claim(case_id: str) -> Optional[Dict[str, Any]]:
    for claim in load_claims():
        if claim.get("case_id") == case_id:
            return claim
    return None


MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _month_number(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    text = _norm_text(value)
    if text in MONTHS:
        return MONTHS[text]
    try:
        number = int(text)
        return number if 1 <= number <= 12 else None
    except (TypeError, ValueError):
        return None


def find_matching_claims(
    party_id: str,
    claim_type: Optional[str] = None,
    claim_month: Optional[str] = None,
    claim_year: Optional[str] = None,
    claim_status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    claims = get_claims_for_party(party_id)
    month_num = _month_number(claim_month)

    result = []
    for claim in claims:
        if claim_type and _norm_text(claim.get("case_type")) != _norm_text(claim_type):
            continue
        if claim_status and _norm_text(claim.get("status")) != _norm_text(claim_status):
            continue

        created = datetime.strptime(claim["created_at"], "%Y-%m-%d")
        if month_num and created.month != month_num:
            continue
        if claim_year:
            try:
                if created.year != int(claim_year):
                    continue
            except ValueError:
                pass
        result.append(claim)

    return result


DOCUMENT_ALIASES = {
    "pathology report": "original pathology report",
    "office note": "treating provider office note",
}


def get_document_context(claim: Dict[str, Any]) -> Dict[str, Any]:
    guidance = load_document_guideline()
    docs = claim.get("documents_needed", []) or []

    doc_guidance: Dict[str, str] = {}
    alt_guidance: Dict[str, str] = {}

    for doc in docs:
        key = DOCUMENT_ALIASES.get(doc.lower(), doc.lower())
        entry = guidance.get("document_guidance", {}).get(key)
        if entry and entry.get("en"):
            doc_guidance[doc] = entry["en"]

        alt = guidance.get("document_alternative_guidance", {}).get(key)
        if alt and alt.get("en"):
            alt_guidance[doc] = alt["en"]

    case_type = claim.get("case_type")
    case_type_guidance = guidance.get("case_type_guidance", {}).get(case_type, {}).get("en")

    return {
        "default_guidance": guidance.get("default_guidance", {}).get("en"),
        "case_type_guidance": case_type_guidance,
        "document_guidance": doc_guidance,
        "document_alternative_guidance": alt_guidance,
        "claim_followup_settings": guidance.get("claim_followup_settings", {}),
        "claim_followup_guidance": guidance.get("claim_followup_guidance", []),
        "claim_followup_fallback": guidance.get("claim_followup_fallback", {}).get("en"),
    }


def get_claim_field_context() -> Dict[str, str]:
    schema = load_claim_schema().get("field_descriptions", {})
    return {name: info.get("description", "") for name, info in schema.items()}


def get_representative_by_name(name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    target = _norm_text(name)
    for rep in load_representatives():
        if _norm_text(rep.get("rep_name")) == target:
            return rep
    return None
