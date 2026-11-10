"""Stage 1 server — bare /ask with typed I/O and real token usage.

Run: uvicorn serve_stage1:app --port 8000 --reload
"""

from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI()
client = Anthropic()


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: str
    tokens_used: int


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question and surface real token usage for cost visibility."""

    message = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": body.question}],
    )

    # The reply is a list of content blocks; the answer lives in the text block.
    answer = next((block.text for block in message.content if block.type == "text"), "")
    tokens_used = message.usage.input_tokens + message.usage.output_tokens

    return AskResponse(answer=answer, tokens_used=tokens_used)
