"""Week 1 live demo — five stages in one file, built up live in class."""

import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = Anthropic()  # Reads ANTHROPIC_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000

# Stage 5 — per-1K-token input/output USD (derived from Anthropic list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (0.004, 0.02),
    "claude-sonnet-5-5": (0.002, 0.01),
    "claude-haiku-5-5": (0.0001, 0.0005),
}


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (input_tokens / 1000 * input_per_1k) + (output_tokens / 1000 * output_per_1k)


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: Anthropic structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    message = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": question}],
        output_format=Answer,
    )

    parsed = message.parsed_output
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    return parsed, input_tokens + output_tokens, input_tokens, output_tokens


def call_model_unsafe(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    # The reply is a list of content blocks; the answer lives in the text block.
    raw = next((block.text for block in message.content if block.type == "text"), "")
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    return answer, input_tokens + output_tokens, input_tokens, output_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question with structured output, guardrails, and cost visibility."""

    model = body.model or DEFAULT_MODEL
    last_error: str | None = None

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            start = time.perf_counter()

            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, input_tokens, output_tokens = call_model_unsafe(
                    body.question, model
                )
            else:
                answer, tokens_used, input_tokens, output_tokens = call_model_structured(
                    body.question, model
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, input_tokens, output_tokens)

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
