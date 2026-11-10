"""Chat view — ask the /ask API questions in a chat-style layout."""

import json
from datetime import datetime

import streamlit as st

from ask_client import MODELS, answer_text, call_ask, explain_error

SUGGESTIONS = [
    "What is RAG in one sentence?",
    "Explain embeddings to a beginner",
    "When should I use a vector database?",
]

st.set_page_config(layout="centered")
messages: list[dict] = st.session_state.setdefault("messages", [])
running_stage = st.session_state.get("running_stage")


def queue_question(question: str) -> None:
    st.session_state["pending_question"] = question


def reply_facts(data: dict) -> str:
    """One quiet line under each answer: model · tokens · cost · time · confidence."""

    answer = data.get("answer")
    details = answer if isinstance(answer, dict) else {}
    facts = []
    if "model" in data:
        facts.append(data["model"])
    if "tokens_used" in data:
        facts.append(f"{data['tokens_used']:,} tokens")
    if "cost_usd" in data:
        facts.append(f"${data['cost_usd']:.4f}")
    if "latency_ms" in data:
        facts.append(f"{data['latency_ms'] / 1000:.1f} s")
    if "confidence" in details:
        facts.append(f"{details['confidence']:.0%} confident")
    if details.get("sources_needed"):
        facts.append("sources recommended")
    return " · ".join(facts)


def source_labels(data: dict) -> str:
    """Session 2: the stored documents the answer cites."""

    answer = data.get("answer")
    citations = answer.get("citations", []) if isinstance(answer, dict) else []
    return ", ".join(f"`{label}`" for label in dict.fromkeys(citations))


def render_reply(index: int, status: int, data: dict | str) -> None:
    if status == 0:
        st.error(data)
        return
    if status != 200 or not isinstance(data, dict):
        st.error(f"**HTTP {status}** — {explain_error(status, data)}")
        return

    st.markdown(answer_text(data))
    st.caption(reply_facts(data))
    if labels := source_labels(data):
        st.caption(f"Cited: {labels}")
    with st.expander("Details"):
        st.json(data)
        st.download_button(
            "Download JSON",
            json.dumps(data, indent=2),
            file_name=f"ask-reply-{index}.json",
            mime="application/json",
            key=f"download_{index}",
            on_click="ignore",
        )


with st.sidebar:
    st.subheader("Chat options")
    if st.button("New chat", icon=":material/edit_square:", use_container_width=True):
        messages.clear()
        st.rerun()
    model = None
    force_bad = False
    if running_stage and running_stage >= 4:
        model = st.selectbox(
            "Model",
            [None, *MODELS],
            format_func=lambda m: m or "Server default",
        )
    if running_stage and running_stage >= 3:
        force_bad = st.toggle(
            "Break first attempt (force_bad)",
            help="The first reply is made invalid so the server's retry has to recover.",
        )

pending = st.session_state.pop("pending_question", None)

if not messages and not pending:
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
    st.markdown(f"<h1 class='greeting'>{greeting}</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='greeting-sub'>Ask a question and your <code>/ask</code> API will answer it.</p>",
        unsafe_allow_html=True,
    )
    for column, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        column.button(
            suggestion,
            on_click=queue_question,
            args=(suggestion,),
            use_container_width=True,
        )

for index, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_reply(index, message["status"], message["data"])

question = st.chat_input("Ask a question…") or pending

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload: dict = {"question": question}
    if force_bad:
        payload["force_bad"] = True
    if model:
        payload["model"] = model

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            status, data = call_ask(st.session_state["base_url"], payload)
        render_reply(len(messages), status, data)
    messages.append({"role": "assistant", "status": status, "data": data})

if messages:
    st.caption("Each question is answered on its own — this API doesn't remember earlier messages.")
