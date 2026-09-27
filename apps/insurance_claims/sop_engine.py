from __future__ import annotations

from typing import Any, Dict, List

from .llm import extract_message, grounded_claim_answer
from .tools import (
    PII_FIELDS,
    find_matching_claims,
    get_claim,
    get_claim_field_context,
    get_document_context,
    identify_policyholder,
)


# ============================================================
# SOP PHASES
# ============================================================

PHASE_VERIFY_ID = "VERIFY_ID"
PHASE_RESOLVE_INTENT = "RESOLVE_INTENT"
PHASE_PROCESS_CASE = "PROCESS_CASE"
PHASE_POST_PROCESS = "POST_PROCESS"

PHASE_COMPLETE = "COMPLETE"
PHASE_ESCALATED = "ESCALATED"


# ============================================================
# STATE
# ============================================================

def new_state() -> Dict[str, Any]:
    return {
        "phase": PHASE_VERIFY_ID,

        "verified": False,
        "party_id": None,

        # Identity information
        "policy_number": None,
        "full_name": None,
        "dob": None,
        "phone": None,
        "email": None,
        "id_last4": None,

        "matched_pii": 0,
        "matched_pii_fields": [],

        # Intent / cross-phase memory
        "intent": None,
        "pending_request": None,

        "claim_type": None,
        "claim_month": None,
        "claim_year": None,
        "claim_status_hint": None,

        "selected_claim_id": None,

        # Emotional support / recovery
        "emotion": "neutral",
        "refusal_count": 0,

        # Out-of-scope handling
        "out_of_scope_count": 0,

        # True when the agent has asked:
        # "Would you like a human representative?"
        "human_offer_pending": False,

        # Case conversation
        "case_notes": [],

        # Post processing
        "email_offered": False,
        "email_sent": False,
        "email_summary": None,

        # Session control
        "ended": False,
    }


# ============================================================
# MEMORY
# ============================================================

def _save_useful_information(
    state: Dict[str, Any],
    extracted: Dict[str, Any],
    user_message: str,
) -> None:
    """
    Save useful information whenever the caller provides it.

    This is important because information may be supplied
    before the SOP phase where it is needed.

    Example:
        During VERIFY_ID:
        "I'm calling about my denied healthcare claim from January."

    We remember that information but do NOT use claim data
    until identity verification succeeds.
    """

    direct_map = {
        "full_name": "full_name",
        "dob": "dob",
        "phone": "phone",
        "email": "email",
        "id_last4": "id_last4",
        "policy_number": "policy_number",

        "claim_type": "claim_type",
        "claim_month": "claim_month",
        "claim_year": "claim_year",
        "claim_status": "claim_status_hint",
    }

    for source, target in direct_map.items():
        value = extracted.get(source)

        if value not in (
            None,
            "",
            "none",
        ):
            state[target] = value

    # Save intent and the ACTUAL original user question.
    intent = extracted.get("intent")

    if intent not in (
        None,
        "",
        "none",
    ):
        state["intent"] = intent
        state["pending_request"] = user_message

    # Save emotion.
    emotion = extracted.get("emotion")

    if emotion:
        state["emotion"] = emotion

    # Track refusal attempts.
    if extracted.get("verification_refusal"):
        state["refusal_count"] += 1


# ============================================================
# EMOTIONAL SUPPORT
# ============================================================

def _empathy_prefix(
    emotion: str,
) -> str:
    mapping = {
        "frustrated":
            "I understand this is frustrating. ",

        "angry":
            "I understand why you're upset. ",

        "anxious":
            "I understand this can feel stressful. ",

        "confused":
            "I understand this can be confusing. ",

        "refusing":
            "I understand you'd rather not repeat information. ",
    }

    return mapping.get(
        emotion,
        "",
    )


# ============================================================
# HELPER: FRIENDLY PII NAMES
# ============================================================

def _format_field_name(
    field: str,
) -> str:
    return {
        "full_name":
            "full name",

        "dob":
            "date of birth",

        "phone":
            "phone number",

        "email":
            "email address",

        "id_last4":
            "last four digits of your SSN or ID",
    }[field]


# ============================================================
# YES / NO PARSING
# ============================================================

def _parse_yes_no(
    user_message: str,
) -> str:
    """
    Deterministic yes/no interpretation.

    Used for:
    - email consent
    - human transfer consent

    Important SOP decisions should not depend entirely
    on LLM classification.
    """

    text = (
        user_message
        .strip()
        .lower()
        .rstrip(".!?")
    )

    yes_phrases = {
        "yes",
        "yes please",
        "yes, please",
        "yeah",
        "yeah please",
        "yep",
        "yep please",
        "sure",
        "sure please",
        "okay",
        "ok",
        "please do",
        "please",
        "that works",
        "do that",
        "connect me",
        "transfer me",
        "send it",
        "send the email",
        "send me the email",
        "send me a summary",
        "send the summary",
        "i would like that",
        "i'd like that",
    }

    no_phrases = {
        "no",
        "no thanks",
        "no thank you",
        "no, thanks",
        "no, thank you",
        "nope",
        "not now",
        "don't",
        "do not",
        "skip it",
        "skip the email",
        "no email",
        "not necessary",
        "i don't want it",
        "i do not want it",
    }

    if text in yes_phrases:
        return "yes"

    if text in no_phrases:
        return "no"

    return "unknown"


# ============================================================
# HUMAN REPRESENTATIVE HELPERS
# ============================================================

def _is_direct_human_request(
    user_message: str,
) -> bool:
    """
    Deterministic backup for explicit human requests.
    """

    text = user_message.lower()

    phrases = [
        "human representative",
        "human agent",
        "real person",
        "real agent",

        "speak to a human",
        "talk to a human",
        "speak with a human",
        "talk with a human",

        "speak to someone",
        "talk to someone",
        "speak with someone",
        "talk with someone",

        "transfer me to a human",
        "transfer me to an agent",

        "let me talk to someone",
        "let me speak to someone",
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def _escalate_to_human(
    state: Dict[str, Any],
) -> str:
    """
    Simulate escalation to a human representative.
    """

    state["phase"] = PHASE_ESCALATED
    state["human_offer_pending"] = False
    state["ended"] = True

    return (
        "Of course. I've marked this conversation for transfer "
        "to a human claims representative. "
        "For this demo, the transfer is simulated."
    )


def _handle_human_offer_response(
    state: Dict[str, Any],
    user_message: str,
):
    """
    Handle a response after the system explicitly offered
    a human representative.
    """

    if not state.get("human_offer_pending"):
        return None

    choice = _parse_yes_no(
        user_message
    )

    if choice == "yes":
        return _escalate_to_human(
            state
        )

    if choice == "no":
        state["human_offer_pending"] = False

        # Give the caller another chance to continue normally.
        state["out_of_scope_count"] = 0
        state["refusal_count"] = 0

        return (
            "No problem. We can continue with the "
            "insurance support workflow."
        )

    # The caller didn't give a clear yes/no.
    # Let their new message proceed normally instead of
    # trapping them in the human-transfer question.
    state["human_offer_pending"] = False
    state["out_of_scope_count"] = 0

    return None


# ============================================================
# OUT-OF-SCOPE HANDLING
# ============================================================

def _handle_out_of_scope(
    state: Dict[str, Any],
) -> str:
    """
    Reject unrelated questions.

    Repeated irrelevant attempts cause an offer
    to speak to a human representative.
    """

    state["out_of_scope_count"] += 1

    if state["out_of_scope_count"] >= 3:
        state["human_offer_pending"] = True

        return (
            "I'm only able to help with insurance claims and "
            "related customer-service questions. "
            "We've gone outside that scope several times. "
            "Would you like me to connect you with a "
            "human claims representative?"
        )

    return (
        "I can help with insurance claims and related "
        "customer-service questions, but I can't answer "
        "unrelated questions. If you'd like, we can "
        "continue with your insurance claim."
    )


# ============================================================
# PHASE 1: VERIFY_ID
# ============================================================

def _handle_verification(
    state: Dict[str, Any],
    extracted: Dict[str, Any],
    user_message: str,
) -> str:
    """
    STRICT VERIFY_ID GATE.

    The caller must match at least 3 PII fields from:

    - Full name
    - DOB
    - Phone
    - Email
    - SSN / ID last four

    Policy number can help locate a record but DOES NOT
    count toward the required 3 PII matches.

    No claim details are disclosed before verification.
    """

    person, match_count, matched_fields = (
        identify_policyholder(
            state
        )
    )

    state["matched_pii"] = match_count
    state["matched_pii_fields"] = matched_fields

    empathy = _empathy_prefix(
        state.get(
            "emotion",
            "neutral",
        )
    )

    # --------------------------------------------------------
    # VERIFICATION SUCCESS
    # --------------------------------------------------------

    if (
        person
        and match_count >= 3
    ):
        state["verified"] = True
        state["party_id"] = person["party_id"]

        state["phase"] = PHASE_RESOLVE_INTENT

        # Only now may we move toward claim access.
        return _resolve_intent(
            state,
            user_message,
            just_verified=True,
        )

    # --------------------------------------------------------
    # EXPLAIN WHY VERIFICATION MATTERS
    # --------------------------------------------------------

    # If the user specifically asked about a denial,
    # make the privacy explanation conversational.
    if state.get("intent") == "denial_question":
        privacy_explanation = (
            "Claim details are protected, so I need to complete "
            "identity verification before I can discuss why the "
            "claim was denied. "
        )

    else:
        privacy_explanation = (
            "Claim details are protected, so I need to complete "
            "identity verification before I can discuss the claim. "
        )

    # --------------------------------------------------------
    # REPEATED REFUSAL
    # --------------------------------------------------------

    if state["refusal_count"] >= 2:
        state["human_offer_pending"] = True

        return (
            empathy
            + privacy_explanation
            + "You can use any three of your full name, date of birth, "
            "phone number, email address, or last four digits of your "
            "SSN/ID. If you'd rather not continue automated "
            "verification, would you like me to connect you with "
            "a human claims representative?"
        )

    supplied = [
        field
        for field in PII_FIELDS
        if state.get(field)
    ]

    missing = [
        field
        for field in PII_FIELDS
        if not state.get(field)
    ]

    # --------------------------------------------------------
    # ACCOUNT / POLICY NOT SUCCESSFULLY IDENTIFIED
    # --------------------------------------------------------

    if (
        person is None
        and state.get("policy_number")
        and len(supplied) >= 1
    ):
        return (
            empathy
            + privacy_explanation
            + "I wasn't able to verify the account from the "
            "information provided. Please check the policy number "
            "and identity details, or provide another accepted "
            "identity field."
        )

    # --------------------------------------------------------
    # 3+ PROVIDED BUT FEWER THAN 3 MATCHED
    # --------------------------------------------------------

    if (
        len(supplied) >= 3
        and match_count < 3
    ):
        alternatives = ", ".join(
            _format_field_name(
                field
            )
            for field in missing[:2]
        )

        extra = (
            f" You can try {alternatives}."
            if alternatives
            else ""
        )

        return (
            empathy
            + privacy_explanation
            + "I couldn't complete identity verification with the "
            "information provided."
            + extra
            + " You can also ask to speak with a "
            "human claims representative."
        )

    # --------------------------------------------------------
    # PARTIAL VERIFICATION
    # --------------------------------------------------------

    need = max(
        0,
        3 - match_count,
    )

    available = [
        _format_field_name(
            field
        )
        for field in missing
    ]

    options = ", ".join(
        available
    )

    # No PII matched yet.
    if match_count == 0:
        return (
            empathy
            + privacy_explanation
            + "I need at least three matching identity items: "
            "full name, date of birth, phone number, email address, "
            "or the last four digits of your SSN/ID. "
            "You may provide them naturally in one message."
        )

    # Some PII matched, but fewer than three.
    return (
        empathy
        + privacy_explanation
        + f"I have {match_count} matching identity "
        + f"item{'s' if match_count != 1 else ''}, "
        + f"and I need {need} more before I can access claim details. "
        + (
            f"You can use: {options}."
            if options
            else
            "Please provide another accepted identity field."
        )
    )


# ============================================================
# PHASE 2: RESOLVE_INTENT
# ============================================================

def _resolve_intent(
    state: Dict[str, Any],
    user_message: str,
    just_verified: bool = False,
) -> str:
    """
    Resolve which claim the caller means.

    This phase is reached only after successful verification.

    It uses remembered information such as:
        healthcare
        denied
        January

    even when that information was supplied during VERIFY_ID.
    """

    if not state.get("verified"):
        state["phase"] = PHASE_VERIFY_ID

        return (
            "I need to complete identity verification "
            "before accessing claim information."
        )

    candidates = find_matching_claims(
        party_id=state["party_id"],

        claim_type=state.get(
            "claim_type"
        ),

        claim_month=state.get(
            "claim_month"
        ),

        claim_year=state.get(
            "claim_year"
        ),

        claim_status=state.get(
            "claim_status_hint"
        ),
    )

    # --------------------------------------------------------
    # NO MATCH
    # --------------------------------------------------------

    if len(candidates) == 0:
        state["phase"] = PHASE_RESOLVE_INTENT

        prefix = (
            "Thanks, your identity is verified. "
            if just_verified
            else ""
        )

        return (
            prefix
            + "I couldn't identify a single matching claim from "
            "the details I have. Please tell me the claim type, "
            "approximate date or month, or status you're asking about."
        )

    # --------------------------------------------------------
    # MULTIPLE POSSIBLE CLAIMS
    # --------------------------------------------------------

    if len(candidates) > 1:
        state["phase"] = PHASE_RESOLVE_INTENT

        choices = "; ".join(
            (
                f"{claim['case_id']} "
                f"({claim['case_type']}, "
                f"{claim['created_at']}, "
                f"{claim['status']})"
            )
            for claim in candidates[:4]
        )

        prefix = (
            "Thanks, your identity is verified. "
            if just_verified
            else ""
        )

        return (
            prefix
            + "I found more than one possible claim: "
            + choices
            + ". Which one do you mean?"
        )

    # --------------------------------------------------------
    # EXACTLY ONE CLAIM
    # --------------------------------------------------------

    claim = candidates[0]

    state["selected_claim_id"] = claim["case_id"]
    state["phase"] = PHASE_PROCESS_CASE

    intent = state.get(
        "intent"
    )

    prefix = (
        "Thanks, your identity is verified. "
        if just_verified
        else ""
    )

    # Claim is known, but caller hasn't asked a question yet.
    if not intent:
        return (
            prefix
            + f"I found the {claim['case_type']} claim "
            + f"from {claim['created_at']} that you mentioned. "
            + "What would you like to know about this claim?"
        )

    # IMPORTANT:
    # Use the remembered ORIGINAL claim question rather than
    # a later verification-only message such as:
    #
    # "My SSN last four is 4472."
    request_to_answer = (
        state.get(
            "pending_request"
        )
        or user_message
    )

    answer = _process_case(
        state,
        request_to_answer,
    )

    return prefix + answer


# ============================================================
# PHASE 3: PROCESS_CASE
# ============================================================

def _process_case(
    state: Dict[str, Any],
    user_message: str,
) -> str:
    """
    Answer claim-specific questions.

    The LLM is allowed to interpret and phrase responses
    naturally, but the answer is grounded in claim/tool data.
    """

    if not state.get("verified"):
        state["phase"] = PHASE_VERIFY_ID

        return (
            "I need to verify your identity before "
            "discussing claim details."
        )

    case_id = state.get(
        "selected_claim_id"
    )

    if not case_id:
        state["phase"] = PHASE_RESOLVE_INTENT

        return _resolve_intent(
            state,
            user_message,
        )

    claim = get_claim(
        case_id
    )

    if not claim:
        state["phase"] = PHASE_RESOLVE_INTENT

        return (
            "I couldn't retrieve that claim. "
            "I can help identify the claim again "
            "or help you move to a human representative."
        )

    document_context = get_document_context(
        claim
    )

    field_context = get_claim_field_context()

    answer = grounded_claim_answer(
        user_message=user_message,
        claim=claim,
        document_context=document_context,
        field_context=field_context,
        conversation_notes=state["case_notes"],
    )

    state["case_notes"].append(
        {
            "user": user_message,
            "assistant": answer,
        }
    )

    # Move to POST_PROCESS after answering.
    #
    # If the caller asks another insurance question instead
    # of answering yes/no, POST_PROCESS sends them back
    # through PROCESS_CASE.
    state["phase"] = PHASE_POST_PROCESS
    state["email_offered"] = True

    return (
        answer
        + "\n\nWould you like an email summary of what we discussed, "
        + "the claim status/outcome, and the main follow-up items "
        + "or next steps?"
    )


# ============================================================
# EMAIL SUMMARY
# ============================================================

def _build_email_summary(
    state: Dict[str, Any],
) -> str:
    """
    Build a summary containing:
    - claim
    - status
    - outcome/reason
    - important documents
    - deadline
    - conversation highlights
    """

    claim = (
        get_claim(
            state["selected_claim_id"]
        )
        if state.get("selected_claim_id")
        else None
    )

    if not claim:
        return (
            "Insurance claims support conversation summary."
        )

    lines: List[str] = [
        "Insurance Claims Support Summary",
        "",
        f"Claim: {claim.get('case_id')}",
        f"Type: {claim.get('case_type')}",
        f"Status: {claim.get('status')}",
    ]

    # Outcome / reason.
    if claim.get("denial_reason"):
        lines.append(
            "Outcome/reason: "
            + claim["denial_reason"]
            + "."
        )

    elif claim.get("summary"):
        lines.append(
            "Outcome: "
            + claim["summary"]
            + "."
        )

    # Follow-up documents.
    if claim.get("documents_needed"):
        lines.append(
            "Requested documents: "
            + ", ".join(
                claim["documents_needed"]
            )
            + "."
        )

    # Deadline.
    if claim.get("appeal_deadline"):
        lines.append(
            "Appeal deadline: "
            + claim["appeal_deadline"]
            + "."
        )

    # Discussion / follow-up highlights.
    if state.get("case_notes"):
        lines.extend(
            [
                "",
                "Discussion highlights:",
            ]
        )

        for note in state["case_notes"][-3:]:
            lines.append(
                "- "
                + note["assistant"]
            )

    return "\n".join(
        lines
    )


# ============================================================
# PHASE 4: POST_PROCESS
# ============================================================

def _handle_post_process(
    state: Dict[str, Any],
    extracted: Dict[str, Any],
    user_message: str,
) -> str:
    """
    POST_PROCESS:

    Caller explicitly chooses whether to receive an
    email summary.

    YES -> simulated email
    NO  -> skip email

    If caller asks another claim question instead,
    process that question first.
    """

    # First use deterministic parsing.
    choice = _parse_yes_no(
        user_message
    )

    # LLM can be fallback for unusual phrasing.
    if choice == "unknown":
        choice = extracted.get(
            "email_choice",
            "unknown",
        )

    # --------------------------------------------------------
    # YES
    # --------------------------------------------------------

    if choice == "yes":
        summary = _build_email_summary(
            state
        )

        state["email_summary"] = summary
        state["email_sent"] = True

        state["phase"] = PHASE_COMPLETE
        state["ended"] = True

        destination = (
            state.get("email")
            or
            "the verified email address on file"
        )

        return (
            "Done. I simulated sending the summary to "
            f"{destination}."
            + "\n\n"
            + summary
        )

    # --------------------------------------------------------
    # NO
    # --------------------------------------------------------

    if choice == "no":
        state["phase"] = PHASE_COMPLETE
        state["ended"] = True

        return (
            "No problem. I won't send an email summary. "
            "Your support session is complete."
        )

    # --------------------------------------------------------
    # ANOTHER CLAIM QUESTION
    # --------------------------------------------------------

    if extracted.get("intent") not in (
        None,
        "",
        "none",
    ):
        state["intent"] = extracted["intent"]

        state["pending_request"] = user_message

        state["phase"] = PHASE_PROCESS_CASE

        return _process_case(
            state,
            user_message,
        )

    return (
        "Would you like the email summary sent? "
        "Please answer yes or no."
    )


# ============================================================
# MAIN SOP CONTROLLER
# ============================================================

def handle_message(
    user_message: str,
    state: Dict[str, Any],
) -> str:
    """
    Main SOP harness.

    STRICT / DETERMINISTIC PYTHON:
        - SOP phase ordering
        - 3-PII verification
        - privacy gate
        - claim access
        - consent
        - escalation
        - human transfer

    FLEXIBLE LLM:
        - natural-language understanding
        - intent extraction
        - emotion detection
        - ambiguity interpretation
        - grounded claim explanations
    """

    # --------------------------------------------------------
    # SESSION ALREADY COMPLETE
    # --------------------------------------------------------

    if state.get("ended"):
        return (
            "This support session is complete. "
            "Please start a new session if you need more help."
        )

    # --------------------------------------------------------
    # RESPONSE TO PREVIOUS HUMAN OFFER
    # --------------------------------------------------------

    human_response = _handle_human_offer_response(
        state,
        user_message,
    )

    if human_response is not None:
        return human_response

    # --------------------------------------------------------
    # DIRECT HUMAN REQUEST
    # --------------------------------------------------------

    if _is_direct_human_request(
        user_message
    ):
        return _escalate_to_human(
            state
        )

    # --------------------------------------------------------
    # LLM LANGUAGE UNDERSTANDING
    # --------------------------------------------------------

    extracted = extract_message(
        user_message,
        state["phase"],
    )

    # Save useful information regardless of current phase.
    _save_useful_information(
        state,
        extracted,
        user_message,
    )

    # LLM also recognizes explicit human requests.
    if extracted.get("human_request"):
        return _escalate_to_human(
            state
        )

    # --------------------------------------------------------
    # REPRESENTATIVE / THIRD-PARTY CALLER
    # --------------------------------------------------------

    # The fixture data does not provide enough information
    # for a safe automated representative authorization flow.
    # Escalate instead of weakening identity verification.
    if (
        extracted.get(
            "caller_is_representative"
        )
        and state["phase"] == PHASE_VERIFY_ID
    ):
        return _escalate_to_human(
            state
        )

    # --------------------------------------------------------
    # OUT-OF-SCOPE
    # --------------------------------------------------------

    if extracted.get(
        "out_of_scope"
    ):
        return _handle_out_of_scope(
            state
        )

    # Caller returned to a valid insurance-support topic.
    # Treat future irrelevant attempts as a new sequence.
    state["out_of_scope_count"] = 0

    # --------------------------------------------------------
    # SOP PHASE ROUTING
    # --------------------------------------------------------

    if state["phase"] == PHASE_VERIFY_ID:
        return _handle_verification(
            state,
            extracted,
            user_message,
        )

    if state["phase"] == PHASE_RESOLVE_INTENT:
        return _resolve_intent(
            state,
            user_message,
        )

    if state["phase"] == PHASE_PROCESS_CASE:
        return _process_case(
            state,
            user_message,
        )

    if state["phase"] == PHASE_POST_PROCESS:
        return _handle_post_process(
            state,
            extracted,
            user_message,
        )

    if state["phase"] == PHASE_ESCALATED:
        return (
            "This conversation has been marked "
            "for a human representative."
        )

    return (
        "This support session is complete."
    )