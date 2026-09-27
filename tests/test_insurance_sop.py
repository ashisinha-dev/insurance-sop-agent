from apps.insurance_claims.sop_engine import new_state
from apps.insurance_claims.tools import (
    find_matching_claims,
    get_claim,
    get_document_context,
    identify_policyholder,
)


def test_margaret_verifies_with_three_pii():
    state = new_state()
    state.update(
        {
            "full_name": "Margaret Chen",
            "dob": "1985-03-15",
            "id_last4": "4472",
            "policy_number": "POL-9921",
        }
    )
    person, count, fields = identify_policyholder(state)
    assert person is not None
    assert person["party_id"] == "P9"
    assert count == 3
    assert set(fields) == {"full_name", "dob", "id_last4"}


def test_two_pii_are_not_enough():
    state = new_state()
    state.update(
        {
            "full_name": "Margaret Chen",
            "dob": "1985-03-15",
            "policy_number": "POL-9921",
        }
    )
    _, count, _ = identify_policyholder(state)
    assert count == 2


def test_policy_number_does_not_count_as_pii():
    state = new_state()
    state.update(
        {
            "full_name": "Margaret Chen",
            "dob": "1985-03-15",
            "policy_number": "POL-9921",
        }
    )
    _, count, fields = identify_policyholder(state)
    assert count == 2
    assert "policy_number" not in fields


def test_margaret_denied_january_healthcare_claim_is_cl_2048():
    matches = find_matching_claims(
        party_id="P9",
        claim_type="healthcare",
        claim_month="January",
        claim_status="denied",
    )
    assert [claim["case_id"] for claim in matches] == ["CL-2048"]


def test_document_guidance_is_grounded_for_cl_2048():
    claim = get_claim("CL-2048")
    context = get_document_context(claim)
    assert "pathology report" in context["document_guidance"]
    assert "office note" in context["document_guidance"]
