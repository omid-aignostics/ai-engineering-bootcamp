"""Session 2 — Pinecone vector store with Pinecone-hosted (integrated) embeddings.

Pinecone turns text into vectors itself, using the model chosen when the index is created, so the
same model embeds documents at ingest time and questions at query time — and no embeddings key is
needed. Everything is configured through environment variables (see .env.example); no secrets here.
The client is created on first use, so /ask keeps working before Pinecone is set up.

Check the connection from a terminal:
  python vector_store.py               # health report
  python vector_store.py create-index  # create the index if it doesn't exist yet, then report
"""

import json
import os
import sys
import time
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from pinecone import EmbedQuery, Pinecone, PineconeError

load_dotenv(Path(__file__).resolve().parent / ".env")

# Fixed when the index is created; Pinecone then uses it for both documents and queries.
EMBEDDING_MODEL = "llama-text-embed-v2"
TEXT_FIELD = "text"  # The document field Pinecone embeds.
DEFAULT_INDEX_NAME = "ask-rag"
DEFAULT_NAMESPACE = "default"  # The documents API needs a non-empty namespace.
UPSERT_BATCH_SIZE = 96  # Pinecone's per-request limit when it embeds the text.


def index_name() -> str:
    return os.environ.get("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)


def namespace() -> str:
    return os.environ.get("PINECONE_NAMESPACE") or DEFAULT_NAMESPACE


def pinecone_configured() -> bool:
    return bool(os.environ.get("PINECONE_API_KEY"))


@cache
def _pinecone() -> Pinecone:
    return Pinecone()  # Reads PINECONE_API_KEY from the environment.


def _index():
    return _pinecone().Index(name=index_name())


def upsert_texts(ids: list[str], texts: list[str], metadata: list[dict] | None = None) -> int:
    """Store texts; Pinecone embeds them on write. Returns the embedding tokens Pinecone reports."""

    metadata = metadata or [{} for _ in texts]
    tokens_used = 0
    for start in range(0, len(texts), UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        response = _index().documents.upsert(
            namespace=namespace(),
            documents=[
                {**extra, "_id": doc_id, TEXT_FIELD: text}
                for doc_id, text, extra in zip(ids[start:end], texts[start:end], metadata[start:end])
            ],
        )
        if response.usage:
            tokens_used += response.usage.embed_total_tokens or 0
    return tokens_used


def query_similar(question: str, top_k: int = 5) -> tuple[list[dict], int]:
    """Find the stored chunks closest to the question. Returns matches plus embedding tokens used."""

    response = _index().documents.search(
        namespace=namespace(),
        top_k=top_k,
        score_by=[EmbedQuery(query=question, field=TEXT_FIELD)],
        include_fields=["*"],
    )
    matches = []
    for doc in response.matches:
        fields = doc.to_dict()
        matches.append(
            {
                "id": doc.id,
                "score": doc.score,
                "text": fields.get(TEXT_FIELD, ""),
                "metadata": {k: v for k, v in fields.items() if k not in {"_id", "_score", TEXT_FIELD}},
            }
        )
    tokens_used = (response.usage.embed_total_tokens or 0) if response.usage else 0
    return matches, tokens_used


def ensure_index() -> str:
    """Create the index (serverless, AWS us-east-1 by default) if it doesn't exist yet."""

    name = index_name()
    if _pinecone().indexes.exists(name):
        return f"Index '{name}' already exists."
    _pinecone().indexes.create(
        name=name,
        schema={"fields": {TEXT_FIELD: {"type": "string", "embed": {"model": EMBEDDING_MODEL}}}},
    )
    return f"Created index '{name}' (Pinecone embeds the '{TEXT_FIELD}' field with {EMBEDDING_MODEL})."


def _index_embedding_model(description) -> str | None:
    """Which hosted model embeds TEXT_FIELD in this index, or None if it isn't set up that way."""

    field = description.schema.fields.get(TEXT_FIELD)
    embed = getattr(field, "embed", None)
    return getattr(embed, "model", None)


def pinecone_health() -> dict:
    """Is Pinecone reachable, and is the index set up for our embedding model? Spends no tokens."""

    report: dict = {
        "ok": False,
        "index_name": index_name(),
        "namespace": namespace(),
        "embedding_model": EMBEDDING_MODEL,
        "pinecone_api_key_set": pinecone_configured(),
    }
    if not report["pinecone_api_key_set"]:
        report["error"] = "PINECONE_API_KEY is not set."
        return report

    start = time.perf_counter()
    try:
        if not _pinecone().indexes.exists(index_name()):
            report["error"] = (
                f"Index '{index_name()}' not found. Create it with "
                "`python vector_store.py create-index`, or set PINECONE_INDEX_NAME."
            )
            return report
        description = _pinecone().indexes.describe(index_name())
        stats = _index().describe_index_stats()
    except PineconeError as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    finally:
        report["latency_ms"] = int((time.perf_counter() - start) * 1000)

    report["index_ready"] = description.status.ready
    report["index_embedding_model"] = _index_embedding_model(description)
    report["model_matches"] = report["index_embedding_model"] == EMBEDDING_MODEL
    report["document_count"] = stats.total_vector_count
    report["namespace_document_count"] = (
        stats.namespaces[namespace()].vector_count if namespace() in stats.namespaces else 0
    )

    problems = []
    if not report["index_ready"]:
        problems.append("index is not ready yet")
    if not report["model_matches"]:
        problems.append(
            f"index field '{TEXT_FIELD}' is embedded by {report['index_embedding_model']}, "
            f"not {EMBEDDING_MODEL} — it was probably created another way; recreate it with "
            "`python vector_store.py create-index` under a new PINECONE_INDEX_NAME"
        )
    if problems:
        report["error"] = "; ".join(problems)
    else:
        report["ok"] = True
    return report


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "create-index":
        try:
            print(ensure_index())
        except PineconeError as exc:
            print(f"Could not create the index: {exc}")
    elif command != "check":
        sys.exit("Usage: python vector_store.py [check | create-index]")
    print(json.dumps(pinecone_health(), indent=2))
