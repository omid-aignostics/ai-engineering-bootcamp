"""Stage 3 server — validation guardrail with one retry and force_bad demo knob.

Run: uvicorn serve_stage3:app --port 8000 --reload
"""

from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI()
client = Anthropic()


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    question: str
    force_bad: bool = False


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int


def call_structured(question: str) -> tuple[Answer, int]:
    message = client.messages.parse(
        model="claude-opus-5-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": question}],
        output_format=Answer,
    )
    parsed = message.parsed_output
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")
    tokens_used = message.usage.input_tokens + message.usage.output_tokens
    return parsed, tokens_used


def call_unsafe(question: str) -> tuple[Answer, int]:
    """Demo path: bad instruction makes confidence a string so validation fails."""
    message = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
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
    raw = next((block.text for block in message.content if block.type == "text"), "")
    answer = Answer.model_validate_json(raw)
    tokens_used = message.usage.input_tokens + message.usage.output_tokens
    return answer, tokens_used


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    last_error: str | None = None

    for attempt in range(2):
        try:
            if body.force_bad and attempt == 0:
                answer, tokens_used = call_unsafe(body.question)
            else:
                answer, tokens_used = call_structured(body.question)
            return AskResponse(answer=answer, tokens_used=tokens_used)
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
