"""Week 1 live demo — five stages in one file, built up live in class.

Session 2 adds RAG: POST /ingest stores documents in Pinecone, and /ask answers from them.
"""

import re
import time
from html import escape
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pinecone import PineconeError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chunking import chunk_text
from vector_store import namespace, pinecone_configured, pinecone_health, query_similar, upsert_texts

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

# Session 2 — grounding: answer only from retrieved chunks, cite them, refuse when they fall short.
REFUSAL = "I don't have enough information to answer that."
GROUNDING_PROMPT = f"""Answer using ONLY the context below.
If the context does not contain the answer, say:
"{REFUSAL}"
Cite the document_id of each chunk you used.

Context:
{{retrieved_chunks}}

Question: {{question}}"""


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    # Session 2 — which context documents the answer used; empty when it refused or had none.
    citations: list[str] = Field(default=[], description="document_id of each context document used")


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    top_k: int = Field(default=5, ge=1, le=20)  # Session 2 — how many stored chunks to retrieve.


class Source(BaseModel):
    """Session 2 — one retrieved chunk the answer could draw on, so callers can cite it."""

    id: str
    document_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    sources: list[Source] = []  # Session 2 — empty when nothing is stored or Pinecone isn't set up.


class IngestRequest(BaseModel):
    """Session 2 — a document to split into chunks and store."""

    # Reject misspelled fields instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    # Required: chunk IDs come from it, so re-ingesting the same document replaces it.
    document_id: str = Field(min_length=1)


class IngestResponse(BaseModel):
    document_id: str
    chunks_ingested: int
    embedding_tokens: int
    namespace: str


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (input_tokens / 1000 * input_per_1k) + (output_tokens / 1000 * output_per_1k)


def retrieve_context(question: str, top_k: int) -> list[Source]:
    """Session 2: the stored chunks most similar to the question (none if Pinecone isn't set up)."""

    if not pinecone_configured():
        return []
    try:
        matches, _ = query_similar(question, top_k=top_k)
    except PineconeError as exc:
        raise HTTPException(status_code=503, detail=f"Could not search the vector store: {exc}") from exc
    return [
        Source(
            id=match["id"],
            document_id=str(match["metadata"].get("document_id", "")),
            score=match["score"],
            text=match["text"],
        )
        for match in matches
    ]


def build_prompt(question: str, sources: list[Source]) -> str:
    """Session 2: wrap the question in the grounding prompt; plain question when Pinecone isn't set up."""

    if not sources:
        return question
    retrieved_chunks = "\n".join(
        f'<document chunk_id="{escape(source.id)}" document_id="{escape(source.document_id)}">\n'
        f"{source.text}\n</document>"
        for source in sources
    )
    return GROUNDING_PROMPT.format(retrieved_chunks=retrieved_chunks, question=question)


def check_citations(answer: Answer, sources: list[Source]) -> None:
    """Session 2 guardrail: a citation must name a document we actually retrieved."""

    unknown = set(answer.citations) - {source.document_id for source in sources}
    if unknown:
        raise ValueError(f"Answer cited documents that were not retrieved: {sorted(unknown)}")


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
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed, citations. "
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

    # Session 2: retrieve once, before the retry loop, so a retry reuses the same context.
    retrieval_start = time.perf_counter()
    sources = retrieve_context(body.question, body.top_k)
    prompt = build_prompt(body.question, sources)
    retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)

    # Session 2: the store is set up but has nothing to ground on — refuse without spending tokens.
    if pinecone_configured() and not sources:
        return AskResponse(
            answer=Answer(answer=REFUSAL, confidence=0.0, sources_needed=True),
            tokens_used=0,
            model=model,
            latency_ms=retrieval_ms,
            cost_usd=0.0,
        )

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            start = time.perf_counter()

            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, input_tokens, output_tokens = call_model_unsafe(prompt, model)
            else:
                answer, tokens_used, input_tokens, output_tokens = call_model_structured(prompt, model)

            if sources:
                check_citations(answer, sources)  # A made-up citation fails validation and triggers the retry.
            else:
                answer.citations = []  # No context was given, so there is nothing to cite.

            latency_ms = retrieval_ms + int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, input_tokens, output_tokens)

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                sources=sources,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Session 2: split a document into chunks and store them in Pinecone, which embeds them."""

    if not pinecone_configured():
        raise HTTPException(status_code=503, detail="PINECONE_API_KEY is not set, so there is nowhere to store documents.")
    chunks = chunk_text(body.text)
    if not chunks:
        raise HTTPException(status_code=422, detail="The text is empty after removing whitespace.")

    # Stable chunk IDs per document: re-ingesting the same document overwrites instead of duplicating.
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", body.document_id).strip("-")[:100] or "doc"
    ids = [f"{prefix}-{i:04d}" for i in range(len(chunks))]
    metadata = [{"document_id": body.document_id, "chunk": i} for i in range(len(chunks))]
    try:
        embedding_tokens = upsert_texts(ids, chunks, metadata)
    except PineconeError as exc:
        raise HTTPException(status_code=503, detail=f"Pinecone rejected the documents: {exc}") from exc

    return IngestResponse(
        document_id=body.document_id,
        chunks_ingested=len(chunks),
        embedding_tokens=embedding_tokens,
        namespace=namespace(),
    )


@app.get("/health/pinecone")
def pinecone_health_check(response: Response) -> dict:
    """Debug check: is Pinecone reachable and the index usable? Spends no tokens."""

    report = pinecone_health()
    if not report["ok"]:
        response.status_code = 503
    return report
