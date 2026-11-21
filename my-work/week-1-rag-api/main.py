"""Week 1 live demo — five stages in one file, built up live in class.

Session 2 adds RAG: POST /ingest stores documents in Pinecone, and /ask answers from them.
Session 3 adds an agent: POST /agent lets a Google ADK agent decide what to do with a message.

Endpoints:
    POST /ask              Answer a question from the stored documents, with citations and cost.
    POST /ingest           Split a document into chunks and store them in Pinecone.
    GET  /health/pinecone  Check the Pinecone connection without spending tokens.
    POST /agent            Session 3: the help desk agent; returns its answer plus steps[].

What runs what (from my-work/week-1-rag-api, after `source .venv/bin/activate`):
    uvicorn main:app --port 8000   API server (this file): /ask, /ingest, /agent, /docs
                                   -> http://127.0.0.1:8000/docs  (Swagger: try every endpoint)
    streamlit run demo_page.py     Chat UI; talks to the API server above, or to Render
                                   -> http://localhost:8501
    adk web --port 8080            ADK's test UI for the agent alone (no API server needed)
                                   -> http://127.0.0.1:8080, pick "helpdesk"
    python run_helpdesk.py         Agent in the terminal: THINK / ACT / OBSERVE / DONE
    python vector_store.py         Pinecone health check (no tokens spent)
    On Render the API server runs with: uvicorn main:app --host 0.0.0.0 --port $PORT
    Its URL is kept private (Maven only); open <render-url>/docs to try it. The free tier
    sleeps after ~15 idle minutes, so the first request can take about a minute.

How a question flows through the app ([LLM] = output can vary between runs,
[fixed] = plain code, same input always gives the same output):

    You type a question
    -> [fixed] chat_page.py          takes your question (Streamlit)
    -> [fixed] call_ask()            sends it to /ask (ask_client.py)
    -> [fixed] ask()                 runs the steps below in a fixed order
       -> [fixed] retrieve_context() -> Pinecone
            [fixed] embeds the question        an AI model, but no randomness: same text -> same vector
            [fixed] finds the closest chunks   pure math (similarity scores)
       -> [fixed] build_prompt()       fills the grounding template
       -> [LLM]   call_model_structured() -> Claude writes the answer   <- the only stochastic step
       -> [fixed] check_citations()    rule check on Claude's output
       -> [fixed] compute_cost_usd()   arithmetic
       -> [fixed] AskResponse          packs the result
    -> [fixed] render_reply()        shows it to you (chat_page.py)

This is a RAG pipeline, not an agent: the code fixes the path, and Claude is called at
exactly one point. Because that is the only step that can vary, its output is checked
(the Answer shape and check_citations) and retried once if it's broken.

Sources vs citations:
    sources           What Pinecone found: the closest chunks to the question, best match first.
    answer.citations  What Claude used: the document_ids its answer is based on (empty if it refused).
    Citations are always a subset of sources; check_citations() enforces that.

What this pipeline can't do, and why Session 3 adds an agent:
    /ask gives exactly two kinds of response: an answer from the documents with citations,
    or the refusal. That is deliberate, because it stops the app from making things up. But a
    refusal is a dead end, and not every request is a document question.
    The Session 3 agent (helpdesk/, POST /agent) sits in front of ask() and decides what kind
    of help a message needs; ask() becomes one of its tools (search_handbook calls it directly,
    not over HTTP). It handles what /ask can't:
        no content in the documents   "Can I bring my dog to work?"  -> HR ticket instead of a dead end
        shouldn't be answered at all   "I want to report my manager"  -> straight to a human (ticket)
    The agent doesn't loosen the grounding rule: ask() still answers only from the documents.

    POST /agent flow:
        -> [fixed] agent()            -> run_agent() (helpdesk/runner.py)
        -> [LLM]   agent decides      search the handbook, open a ticket, or both
        -> [fixed] search_handbook()  -> ask() above, the whole RAG pipeline
        -> [LLM]   agent writes the final answer from what the tools returned
        -> [fixed] AgentResponse      answer + steps[] (think, tool, args, observation)

Pinecone details live in vector_store.py; text splitting lives in chunking.py.
"""

import re
import time
from html import escape
from pathlib import Path

from anthropic import Anthropic, APIError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pinecone import PineconeError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chunking import chunk_text
from helpdesk.runner import run_agent
from vector_store import namespace, pinecone_configured, pinecone_health, query_similar, upsert_texts

# ---------------------------------------------------------------------------
# 1. Setup — runs once, when the server starts.
# ---------------------------------------------------------------------------

# Load .env from this folder so the key is found regardless of shell working directory.
# On Render the keys come from the service's Environment settings instead.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = Anthropic()  # Reads ANTHROPIC_API_KEY from the environment; never hardcode keys.

# ---------------------------------------------------------------------------
# 2. Settings
# ---------------------------------------------------------------------------

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000  # Upper limit on the length of Claude's reply.

# Stage 5 — per-1K-token input/output USD (derived from Anthropic list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (0.004, 0.02),
    "claude-sonnet-5-5": (0.002, 0.01),
    "claude-haiku-5-5": (0.0001, 0.0005),
}

# ---------------------------------------------------------------------------
# 3. Grounding prompt (Session 2)
# ---------------------------------------------------------------------------

# Session 2 — grounding: answer only from retrieved chunks, cite them, refuse when they fall short.
# The two {placeholders} are filled per request by build_prompt().
REFUSAL = "I don't have enough information to answer that."
GROUNDING_PROMPT = f"""Answer using ONLY the context below.
If the context does not contain the answer, say:
"{REFUSAL}"
Cite the document_id of each chunk you used.

Context:
{{retrieved_chunks}}

Question: {{question}}"""

# ---------------------------------------------------------------------------
# 4. Data shapes — FastAPI and Pydantic reject anything that doesn't fit these.
# ---------------------------------------------------------------------------


# Claude must reply in exactly the Answer shape (see call_model_structured). Its docstring is
# part of the JSON schema Claude receives, so field notes live in these comments instead:
#   answer          The answer text, or REFUSAL when the context doesn't cover the question.
#   confidence      How sure the model is, from 0 to 1.
#   sources_needed  Session 1 field: whether the answer would benefit from sources.
#   citations       Session 2: the document_ids Claude says it used, picked from the retrieved
#                   chunks in the prompt (a subset of AskResponse.sources). Verified by
#                   check_citations(); empty when the answer is a refusal.
class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    # Session 2 — which context documents the answer used; empty when it refused or had none.
    citations: list[str] = Field(default=[], description="document_id of each context document used")


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens.

    Only `question` is required; the rest are optional knobs for the demos.
    """

    # The examples fill Swagger's "Try it out" box with working values instead of "string".
    question: str = Field(examples=["Do unused vacation days carry over to next year?"])
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    # Stage 4 — optional override to swap models live; must be a key of MODEL_PRICES_PER_1K.
    model: str | None = Field(default=None, examples=["claude-haiku-5-5"])
    top_k: int = Field(default=5, ge=1, le=20)  # Session 2 — how many stored chunks to retrieve.


class Source(BaseModel):
    """Session 2 — one retrieved chunk the answer could draw on, so callers can cite it.

        id           The chunk's ID in Pinecone, e.g. "handbook-0000".
        document_id  The document the chunk came from, e.g. "handbook".
        score        Similarity to the question; higher means more relevant.
        text         The chunk's text, exactly as it was given to Claude.
    """

    id: str
    document_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back.

    tokens_used and cost_usd count Claude's tokens only (Pinecone's embedding is billed
    separately).

    sources vs citations:
        sources           Every chunk Pinecone retrieved, best match first — useful or not.
        answer.citations  Only the document_ids Claude actually used for this answer,
                          always a subset of the document_ids in sources.
    """

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int  # Retrieval plus the successful Claude call.
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
    """Session 2 — what /ingest stored: how many chunks, the embedding tokens, and where."""

    document_id: str
    chunks_ingested: int
    embedding_tokens: int
    namespace: str


class AgentRequest(BaseModel):
    """Session 3 — a message for the help desk agent (a question or a request)."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1, max_length=2000, examples=["Can I bring my dog to the office?"]
    )


class AgentStep(BaseModel):
    """Session 3 — one tool call the agent made: Think -> Act -> Observe.

        think        Why the agent chose this tool, in its own words (can be empty).
        tool         The tool it called, e.g. "search_handbook".
        args         The arguments it passed.
        observation  What the tool returned, cut to 300 characters.
    """

    think: str | None = None
    tool: str
    args: dict
    observation: str | None = None


class AgentResponse(BaseModel):
    """Session 3 — the agent's final answer and the steps it took to get there.

    Only these fields are returned: never API keys, environment values or raw errors.
    """

    answer: str
    steps: list[AgentStep]
    model: str  # The model the agent thinks with (search_handbook's /ask uses DEFAULT_MODEL).
    latency_ms: int
    step_limit_reached: bool  # True if the run was stopped at MAX_STEPS model calls.
    problem: str | None = None  # Why the agent gave no real answer (None when it did). No secrets.


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost.

    A model missing from MODEL_PRICES_PER_1K is priced like DEFAULT_MODEL.
    """

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (input_tokens / 1000 * input_per_1k) + (output_tokens / 1000 * output_per_1k)


def retrieve_context(question: str, top_k: int) -> list[Source]:
    """Session 2: the stored chunks most similar to the question (none if Pinecone isn't set up).

    Pinecone embeds the question with the same model used at ingest and returns the top_k
    closest chunks, best match first.

    Raises:
        HTTPException 503: Pinecone is configured but the search failed.
    """

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
    """Session 2: wrap the question in the grounding prompt; plain question when Pinecone isn't set up.

    Each chunk becomes a <document> block labelled with its chunk_id and document_id,
    so Claude has something concrete to cite.
    """

    if not sources:
        return question
    retrieved_chunks = "\n".join(
        f'<document chunk_id="{escape(source.id)}" document_id="{escape(source.document_id)}">\n'
        f"{source.text}\n</document>"
        for source in sources
    )
    return GROUNDING_PROMPT.format(retrieved_chunks=retrieved_chunks, question=question)


def check_citations(answer: Answer, sources: list[Source]) -> None:
    """Session 2 guardrail: a citation must name a document we actually retrieved.

    Raises:
        ValueError: the answer cites an unknown document. /ask treats this like any
            other invalid answer and retries once.
    """

    unknown = set(answer.citations) - {source.document_id for source in sources}
    if unknown:
        raise ValueError(f"Answer cited documents that were not retrieved: {sorted(unknown)}")


# ---------------------------------------------------------------------------
# 6. Calling Claude
# ---------------------------------------------------------------------------


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: Anthropic structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.

    `question` is the full prompt: the grounding prompt when there is context, otherwise
    the bare question. Returns (answer, total_tokens, input_tokens, output_tokens).

    Raises:
        ValueError: Claude returned nothing that fits the Answer shape.
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

    Used only on the first attempt when force_bad is set, so the retry in /ask has
    something to recover from. Same return value as call_model_structured.

    Raises:
        ValidationError: the reply doesn't fit the Answer shape (the expected outcome).
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


# ---------------------------------------------------------------------------
# 7. Endpoints
# ---------------------------------------------------------------------------


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question with structured output, guardrails, and cost visibility.

    Steps:
        1. Pick the model: the request's, or DEFAULT_MODEL.
        2. Retrieve the top_k closest chunks and build the grounding prompt (once).
        3. Pinecone is set up but empty: return REFUSAL without calling Claude.
        4. Call Claude (the unsafe path first if force_bad), check the citations,
           work out latency and cost, and return.
        5. Invalid answer (wrong shape or made-up citation): retry once.
        6. Both attempts invalid: 502, never a half-parsed answer.
    """

    model = body.model or DEFAULT_MODEL
    # An unknown model name (e.g. Swagger's "string") is the caller's mistake: say so (422)
    # instead of letting Anthropic's "model not found" crash the request with a 500.
    if model not in MODEL_PRICES_PER_1K:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model {model!r}. Use one of: {', '.join(MODEL_PRICES_PER_1K)}.",
        )
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
        except APIError as exc:
            # Claude itself failed (overloaded, rate limit, network). Report the type only.
            raise HTTPException(
                status_code=502, detail=f"The Claude request failed ({type(exc).__name__})."
            ) from exc

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Session 2: split a document into chunks and store them in Pinecone, which embeds them.

    Steps:
        1. No Pinecone key: 503.
        2. Split the text into ~1,000-character chunks (see chunking.py); empty text: 422.
        3. Give each chunk a stable ID, e.g. "handbook-0000", so re-ingesting replaces it.
        4. Store the chunks; Pinecone creates the embeddings. Pinecone error: 503.
    """

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
    """Debug check: is Pinecone reachable and the index usable? Spends no tokens.

    Returns 200 when everything is fine, 503 with an "error" field when not.
    See vector_store.pinecone_health() for what is checked.
    """

    report = pinecone_health()
    if not report["ok"]:
        response.status_code = 503
    return report


@app.post("/agent")
async def agent(body: AgentRequest) -> AgentResponse:
    """Session 3 — let the help desk agent handle a message (see helpdesk/agent.py).

    The agent may answer from the documents (its search_handbook tool calls ask() above),
    open an HR ticket, or both in sequence. Each request is handled on its own, like /ask.

    Steps:
        1. run_agent() runs the agent loop, at most MAX_STEPS model calls.
        2. Return the final answer and one AgentStep per tool call.
        3. Any unexpected failure: 502 with only the error type, never its details.
    """

    start = time.perf_counter()
    try:
        result = await run_agent(body.message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"The agent failed ({type(exc).__name__}).") from exc

    answer = result["answer"]
    if result["step_limit_reached"] and not answer:
        answer = "Stopped: the step limit was reached before a final answer."
    elif not answer:
        answer = "Sorry, I couldn't produce an answer this time. Please try again."

    return AgentResponse(
        answer=answer,
        steps=result["steps"],
        model=result["model"],
        latency_ms=int((time.perf_counter() - start) * 1000),
        step_limit_reached=result["step_limit_reached"],
        problem=result["problem"],
    )
