"""Stage 2 server — structured output turns the chatbot into a component.

Run: uvicorn serve_stage2:app --port 8000 --reload
"""

from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI()
client = Anthropic()


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Force the model into a fixed schema via Anthropic structured output."""

    message = client.messages.parse(
        model="claude-opus-5-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": body.question}],
        output_format=Answer,
    )

    parsed = message.parsed_output
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    tokens_used = message.usage.input_tokens + message.usage.output_tokens
    return AskResponse(answer=parsed, tokens_used=tokens_used)
