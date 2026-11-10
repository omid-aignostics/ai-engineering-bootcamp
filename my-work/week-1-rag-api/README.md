# Week 1 — `/ask` Demo (5 stages)

Build a typed LLM endpoint step by step. Each stage is a standalone FastAPI app you can run and compare.

## Setup

```bash
cp .env.example .env          # ANTHROPIC_API_KEY=sk-ant-...
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo stages

| Stage | File | What you learn |
|-------|------|----------------|
| 1 | `serve_stage1.py` | Bare `/ask` — string answer + `tokens_used` |
| 2 | `serve_stage2.py` | Structured output via Pydantic + `messages.parse` |
| 3 | `serve_stage3.py` | Validation guardrail + retry (`force_bad` demo knob) |
| 4 | `serve_stage4.py` | Per-request `model` override + `latency_ms` |
| 5 | `serve_stage5.py` / `main.py` | Full system + `cost_usd` readout |

Run one stage at a time (only one server on port 8000):

```bash
uvicorn serve_stage1:app --host 127.0.0.1 --port 8000 --reload
# or the full system:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Streamlit demo runner

Interactive UI for all five stages:

```bash
streamlit run demo_page.py
```

Open http://localhost:8501. Set **API base URL** to `http://127.0.0.1:8000` and start the matching stage server in another terminal, or paste your hosted URL (e.g. Render). The sidebar shows which stage the server at that address is running.

To open the page with your hosted URL already filled in:

```bash
DEMO_API_URL=https://your-service.onrender.com streamlit run demo_page.py
```

## Test with curl

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG in one sentence?"}'
```

Stage 5 example (model + cost):

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "model": "claude-haiku-5-5"}'
```

## Session 2 — RAG with Pinecone

`main.py` can answer from your own documents. Pinecone stores them and embeds them itself (`llama-text-embed-v2`), so the only extra key is `PINECONE_API_KEY` (see `.env.example`).

```bash
python vector_store.py create-index   # one time
python vector_store.py                # health report — look for "ok": true
```

Store a document, then ask about it:

```bash
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "The office dog is called Pixel and loves carrots.", "document_id": "handbook"}'

curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the office dog called?"}'
```

`/ask` retrieves the top 5 chunks (`top_k`) and tells Claude to answer **only** from them, citing each `document_id` it used in `answer.citations`, and to reply "I don't have enough information to answer that." when they don't cover the question. `sources` lists every retrieved chunk with its `id` and score. A citation that names a document that wasn't retrieved fails validation and triggers the retry. Without `PINECONE_API_KEY` it answers as before, with `sources` and `citations` empty. `GET /health/pinecone` checks the connection without spending tokens.

## Smoke-test all stages

Requires `.venv` and a valid `ANTHROPIC_API_KEY`:

```bash
python test_all_stages.py
```

## Project layout

```
week-1/
├── main.py              # Full system (stages 1–5 combined)
├── serve_stage1.py … serve_stage5.py
├── demo_page.py         # Streamlit test UI
├── test_all_stages.py   # Automated stage smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```
