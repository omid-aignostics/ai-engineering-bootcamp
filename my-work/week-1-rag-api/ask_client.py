"""Shared helpers for the demo UI pages: talk to /ask and explain what came back."""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx
import streamlit as st

# Show the real folder path, with the home folder shortened to ~.
WORKDIR_CMD = str(Path(__file__).resolve().parent).replace(str(Path.home()), "~", 1)
LOCAL_URL = "http://127.0.0.1:8000"
DEFAULT_URL = os.environ.get("DEMO_API_URL", LOCAL_URL)
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]


def normalize_url(raw: str) -> str:
    """Accept what people paste: no scheme, a trailing slash, or a trailing /ask."""

    url = raw.strip().rstrip("/")
    if url.endswith("/ask"):
        url = url[: -len("/ask")]
    if url and "://" not in url:
        scheme = "http" if url.startswith(("127.0.0.1", "localhost")) else "https"
        url = f"{scheme}://{url}"
    return url


def is_local(url: str) -> bool:
    return urlparse(url).hostname in {"127.0.0.1", "localhost", "0.0.0.0"}


@st.cache_data(ttl=30, show_spinner=False)
def detect_stage(base_url: str) -> int | None:
    """Read the server's OpenAPI schema to see which stage it runs — costs no tokens.

    Returns the stage number, 0 for "reachable but not a week-1 server", or None if unreachable.
    """

    try:
        spec = httpx.get(f"{base_url}/openapi.json", timeout=90.0).json()
    except (httpx.HTTPError, ValueError):
        return None

    schemas = spec.get("components", {}).get("schemas", {})
    request_fields = schemas.get("AskRequest", {}).get("properties", {})
    response_fields = schemas.get("AskResponse", {}).get("properties", {})
    if "/ask" not in spec.get("paths", {}) or not response_fields:
        return 0
    if "cost_usd" in response_fields:
        return 5
    if "model" in response_fields:
        return 4
    if "force_bad" in request_fields:
        return 3
    if response_fields.get("answer", {}).get("type") == "string":
        return 1
    return 2


def call_ask(base_url: str, payload: dict) -> tuple[int, dict | str]:
    """POST to /ask. Status 0 means no HTTP reply; the data is then a friendly message."""

    try:
        response = httpx.post(f"{base_url}/ask", json=payload, timeout=120.0)
    except httpx.ConnectError:
        if is_local(base_url):
            return 0, (
                f"Nothing is running at {base_url}. Start the API server first, e.g. "
                "`uvicorn main:app --port 8000 --reload`."
            )
        return 0, (
            f"Could not reach {base_url}. Check the address, and make sure the "
            "Render service is not suspended."
        )
    except httpx.TimeoutException:
        return 0, (
            "No reply after 2 minutes. A sleeping Render service can take up to a "
            "minute to wake up — try again."
        )
    except httpx.HTTPError as exc:
        return 0, f"Request failed: {exc}"

    try:
        return response.status_code, response.json()
    except json.JSONDecodeError:
        return response.status_code, response.text


def explain_error(status: int, data: dict | str) -> str:
    """Turn an HTTP error into a plain-English hint."""

    detail = data.get("detail", data) if isinstance(data, dict) else data
    text = detail if isinstance(detail, str) else json.dumps(detail)
    if "openai" in text.lower():
        return (
            "This server is running OpenAI code, so it is not the week-1 app — probably a "
            "different project is using this port. Stop it and start the week-1 server."
        )
    if status == 404:
        return "This server has no /ask endpoint. Check the API base URL in the sidebar."
    if status == 422:
        return "The server did not accept the request fields. It may be running a different stage."
    if status == 502:
        return "The model's reply did not match the expected format, even after one retry. Try again."
    if status >= 500:
        return (
            "The server hit an error. Check its logs (your terminal, or the Logs tab on Render). "
            "The most common cause is a missing or wrong ANTHROPIC_API_KEY."
        )
    return f"The server returned HTTP {status}."


def answer_text(data: dict) -> str:
    """Stage 1 returns the answer as a string; later stages wrap it in an object."""

    answer = data.get("answer", "")
    return answer.get("answer", "") if isinstance(answer, dict) else answer


def collect_metrics(data: dict) -> list[tuple[str, str]]:
    """Pick out whichever numbers this stage returns, formatted for display."""

    answer = data.get("answer")
    details = answer if isinstance(answer, dict) else {}
    metrics: list[tuple[str, str]] = []
    if "tokens_used" in data:
        metrics.append(("Tokens used", f"{data['tokens_used']:,}"))
    if "cost_usd" in data:
        metrics.append(("Cost (USD)", f"${data['cost_usd']:.6f}"))
    if "latency_ms" in data:
        metrics.append(("Response time", f"{data['latency_ms'] / 1000:.1f} s"))
    if "confidence" in details:
        metrics.append(("Confidence", f"{details['confidence']:.0%}"))
    if "sources_needed" in details:
        metrics.append(("Needs sources", "Yes" if details["sources_needed"] else "No"))
    return metrics
