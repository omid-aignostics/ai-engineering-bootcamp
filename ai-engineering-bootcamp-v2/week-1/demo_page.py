"""Week 1 /ask demo UI — a chat view plus the five-stage demo runner.

Run this page:
  streamlit run demo_page.py

Start with your hosted API already filled in:
  DEMO_API_URL=https://your-service.onrender.com streamlit run demo_page.py

Colors and fonts live in .streamlit/config.toml.
"""

import streamlit as st

from ask_client import DEFAULT_URL, WORKDIR_CMD, detect_stage, is_local, normalize_url

# Chat-app touches the theme can't express: serif replies, soft user bubbles, a centered greeting.
STYLE = """
<style>
[data-testid="stChatMessage"] {
    padding: 0.75rem 1rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: rgba(128, 112, 96, 0.10);
    border-radius: 1rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background-color: transparent;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stMarkdownContainer"] p {
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 1.06rem;
    line-height: 1.7;
}
[data-testid="stChatInput"] > div {
    border-radius: 1.25rem;
}
.greeting {
    text-align: center;
    font-weight: 500;
    margin-top: 14vh;
}
.greeting-sub {
    text-align: center;
    opacity: 0.7;
    margin-bottom: 2rem;
}
</style>
"""

st.set_page_config(page_title="Week 1 /ask Demo", page_icon="💬")
st.markdown(STYLE, unsafe_allow_html=True)

pages = st.navigation(
    [
        st.Page("chat_page.py", title="Chat", icon=":material/chat_bubble:", default=True),
        st.Page("stage_runner.py", title="Stage demos", icon=":material/science:"),
    ]
)

with st.sidebar:
    with st.expander("API connection", icon=":material/link:"):
        base_url = normalize_url(
            st.text_input(
                "API base URL",
                DEFAULT_URL,
                key="api_url",
                help="Your Render URL (https://…onrender.com), or http://127.0.0.1:8000 for a "
                "server on this computer. No need to add /ask.",
            )
        )
        if st.button("Check again", use_container_width=True):
            detect_stage.clear()
            st.rerun()
        st.caption(f"Start this page: `cd {WORKDIR_CMD} && streamlit run demo_page.py`")

    local = is_local(base_url)
    with st.spinner("Checking your API…"):
        running_stage = detect_stage(base_url)
    where = "this computer" if local else "hosted"
    if running_stage is None:
        st.error(f"Can't reach the API ({where}). Open **API connection** to fix the address.")
    elif running_stage == 0:
        st.warning("Something answers here, but it isn't a week-1 stage server.")
    else:
        st.success(f"Connected · {where} · stage {running_stage}")

st.session_state["base_url"] = base_url
st.session_state["is_local"] = local
st.session_state["running_stage"] = running_stage

pages.run()
