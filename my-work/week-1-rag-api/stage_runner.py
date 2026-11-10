"""Stage demos — one tab per /ask stage, with what to look for in each reply."""

import json
import shlex

import streamlit as st

from ask_client import MODELS, WORKDIR_CMD, call_ask, collect_metrics, explain_error

STAGES = [
    {
        "num": 1,
        "title": "Bare /ask",
        "serve": "uvicorn serve_stage1:app --port 8000 --reload",
        "look_for": "Plain `answer` string and real `tokens_used`.",
        "dummy_question": "What is Retrieval-Augmented Generation in one sentence?",
        "fields": [],
    },
    {
        "num": 2,
        "title": "Structured output",
        "serve": "uvicorn serve_stage2:app --port 8000 --reload",
        "look_for": "`answer` is an object with `confidence` and `sources_needed`.",
        "dummy_question": "Explain what an embedding is in one sentence.",
        "fields": [],
    },
    {
        "num": 3,
        "title": "Guardrail + retry",
        "serve": "uvicorn serve_stage3:app --port 8000 --reload",
        "look_for": "Normal question works; `force_bad` triggers retry then succeeds.",
        "dummy_question": "What is a vector database?",
        "dummy_force_bad": True,
        "fields": ["force_bad"],
    },
    {
        "num": 4,
        "title": "Model selectable",
        "serve": "uvicorn serve_stage4:app --port 8000 --reload",
        "look_for": "`model` and `latency_ms` in the response; swap models live.",
        "dummy_question": "What is chunking in RAG?",
        "dummy_model": "claude-haiku-5-5",
        "fields": ["force_bad", "model"],
    },
    {
        "num": 5,
        "title": "Cost readout",
        "serve": "uvicorn serve_stage5:app --port 8000 --reload",
        "look_for": "`cost_usd` closes the loop — same prompt, different model, different cost.",
        "dummy_question": "What is Retrieval-Augmented Generation in one sentence?",
        "dummy_model": "claude-opus-5-5",
        "fields": ["force_bad", "model"],
    },
]

st.set_page_config(layout="wide")
base_url: str = st.session_state["base_url"]
local: bool = st.session_state["is_local"]
running_stage: int | None = st.session_state.get("running_stage")


def build_payload(
    question: str,
    stage: dict,
    force_bad: bool,
    model: str | None,
) -> dict:
    payload: dict = {"question": question}
    if "force_bad" in stage["fields"]:
        payload["force_bad"] = force_bad
    if "model" in stage["fields"] and model:
        payload["model"] = model
    return payload


def render_curl(payload: dict) -> str:
    return (
        f"curl -s -X POST {base_url}/ask "
        f'-H "Content-Type: application/json" -d {shlex.quote(json.dumps(payload))}'
    )


def render_result(stage_num: int, status: int, data: dict | str) -> None:
    if status == 0:
        st.error(data)
        return
    if status != 200 or not isinstance(data, dict):
        st.error(f"**HTTP {status}** — {explain_error(status, data)}")
        with st.expander("Error details"):
            if isinstance(data, dict):
                st.json(data)
            else:
                st.code(str(data))
        return

    answer = data.get("answer")
    with st.container(border=True):
        st.markdown("**Answer**")
        st.write(answer.get("answer", "") if isinstance(answer, dict) else answer)

    metrics = collect_metrics(data)
    if metrics:
        for column, (label, value) in zip(st.columns(len(metrics)), metrics):
            column.metric(label, value)
    if "model" in data:
        st.caption(f"Answered by `{data['model']}`")

    st.download_button(
        "Download JSON",
        json.dumps(data, indent=2),
        file_name=f"ask-stage{stage_num}.json",
        mime="application/json",
        key=f"download_stage_{stage_num}",
        on_click="ignore",
    )
    with st.expander("Raw JSON response"):
        st.json(data)


st.title("Stage demos")
st.caption("Pick a demo tab, type a question, and press **Ask**.")

tabs = st.tabs([f"Demo {s['num']}: {s['title']}" for s in STAGES])

for tab, stage in zip(tabs, STAGES):
    with tab:
        num = stage["num"]
        st.subheader(f"Demo {num} — {stage['title']}")
        st.markdown(f"**Look for:** {stage['look_for']}")
        if running_stage and running_stage != num:
            st.info(
                f"The server at this address runs stage {running_stage}, so the reply will "
                f"look like stage {running_stage}, not stage {num}."
            )

        with st.form(key=f"form_{num}"):
            question = st.text_input(
                "Question",
                stage["dummy_question"],
                key=f"q_{num}",
                placeholder="Type a question to send to /ask…",
            )
            force_bad = stage.get("dummy_force_bad", False)
            model = stage.get("dummy_model")
            if "force_bad" in stage["fields"]:
                force_bad = st.checkbox(
                    "Break the first attempt on purpose (force_bad)",
                    value=stage.get("dummy_force_bad", False),
                    key=f"bad_{num}",
                    help="The first reply is made invalid so you can watch the retry recover.",
                )
            if "model" in stage["fields"]:
                options = [None, *MODELS]
                model = st.selectbox(
                    "Model",
                    options,
                    index=options.index(model) if model in options else 0,
                    format_func=lambda m: m or "Server default",
                    key=f"model_{num}",
                )
            submitted = st.form_submit_button("Ask", type="primary")

        payload = build_payload(question, stage, force_bad, model)
        result_key = f"result_{num}"
        if submitted:
            if not question.strip():
                st.warning("Type a question first.")
            else:
                with st.spinner("Asking… a sleeping Render service can take up to a minute to wake up."):
                    st.session_state[result_key] = call_ask(base_url, payload)
        if result_key in st.session_state:
            render_result(num, *st.session_state[result_key])

        if local:
            with st.expander("Run this stage on your computer"):
                st.markdown("**Terminal 1 — start the server** (leave it running):")
                st.code(f"cd {WORKDIR_CMD}\nsource .venv/bin/activate\n{stage['serve']}", language="bash")
                st.markdown("**Terminal 2 — optional: test it without this page:**")
                st.code(render_curl(payload), language="bash")
        else:
            with st.expander("Test this from a terminal instead"):
                st.code(render_curl(payload), language="bash")
