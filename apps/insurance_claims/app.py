from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.insurance_claims.llm import ModelConfigError
from apps.insurance_claims.sop_engine import handle_message, new_state


st.set_page_config(page_title="Insurance Claims SOP Agent", page_icon="🛡️", layout="wide")

if "agent_state" not in st.session_state:
    st.session_state.agent_state = new_state()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. I'm the insurance claims support assistant. "
                "I can help with claim verification, status, denials, documents, payments, and next steps."
            ),
        }
    ]

st.title("Insurance Claims Support Agent")
st.caption("SOP-controlled workflow with natural-language understanding")

with st.sidebar:
    st.subheader("SOP Debug View")
    state = st.session_state.agent_state
    st.write(f"**Phase:** {state['phase']}")
    st.write(f"**Verified:** {'Yes' if state['verified'] else 'No'}")
    st.write(f"**Matched PII:** {state['matched_pii']} / 3 required")
    st.write(f"**Remembered intent:** {state.get('intent') or 'None'}")
    hints = [
        state.get("claim_type"),
        state.get("claim_month"),
        state.get("claim_year"),
        state.get("claim_status_hint"),
    ]
    st.write("**Claim hints:** " + (" / ".join(str(x) for x in hints if x) or "None"))
    st.write(f"**Selected claim:** {state.get('selected_claim_id') or 'None'}")
    st.write(f"**Out-of-scope count:** {state.get('out_of_scope_count', 0)}")
    st.write(f"**Verification refusals:** {state.get('refusal_count', 0)}")

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.agent_state = new_state()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello. I'm the insurance claims support assistant. How can I help you today?",
            }
        ]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input("Type your message")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    try:
        reply = handle_message(user_message, st.session_state.agent_state)
    except ModelConfigError as exc:
        reply = (
            f"Configuration error: {exc} Add MODEL_API_KEY and MODEL_NAME to your .env file, "
            "then restart the app."
        )
    except Exception as exc:
        reply = (
            "I'm having trouble processing that request right now. "
            "Please try again or ask for a human representative.\n\n"
            f"Developer detail: `{type(exc).__name__}`"
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
