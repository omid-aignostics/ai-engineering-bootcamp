"""Employee help desk agent (Session 3) — one agent, one multi-step job.

The job, in one sentence:
    Answer an employee's policy question from the company documents with citations,
    and if the documents can't answer it (or it's sensitive), open an HR ticket instead.

Why this needs an agent and not just /ask:
    /ask can only answer or refuse, and a refusal is a dead end. The agent decides what to
    do next: answer with citations, open a ticket, or send a sensitive topic straight to HR.

How a message flows ([LLM] = output can vary between runs, [fixed] = plain code):
    POST /agent {"message": "..."}       (or: python run_helpdesk.py, or: adk web)
    -> [fixed] run_agent()               helpdesk/runner.py starts the agent loop
    -> [LLM]   agent decides what to do  (THINK)
    -> [fixed] search_handbook()         the real tool (ACT)
         -> [fixed] ask()                the same function behind POST /ask (main.py)
              -> [fixed] retrieve_context() -> Pinecone
              -> [LLM]   Claude answers from the chunks only, with citations
    -> [fixed] search_handbook() returns {found, answer, citations}   (OBSERVE)
    -> [LLM]   agent decides: reply with citations, or call create_hr_ticket()
    -> [fixed] run_agent() returns the final answer plus steps[]

Why search_handbook() calls ask() directly instead of sending an HTTP request to /ask:
    The agent and /ask now run in the same server. A web request to itself would be slower,
    and a busy server could end up waiting on itself. Calling the function gives exactly the
    same result (same retrieval, grounding prompt, citation check and retry), without the web.

Tools:
    search_handbook   REAL: runs the week 1 RAG pipeline (Pinecone + grounded Claude answer).
    create_hr_ticket  PLACEHOLDER: returns a made-up ticket; a real one would call a ticket API.
    Tools never raise: on failure they return {"error": ...}, so Claude sees the problem as an
    observation and tells the employee, instead of the whole run crashing.

Safety limits:
    MAX_STEPS        ADK stops the run after this many model calls, so it can't loop forever.
    Uncited answers  search_handbook passes on only the refusal if /ask gave no citations.

Claude, not Gemini:
    ADK's built-in AnthropicLlm connector calls the Anthropic API directly with
    ANTHROPIC_API_KEY. (We first used LiteLLM for this; dropping it saved about 190 MB of memory.)

Patterns copied from the course template ../week-3-agents/demo1_routing.py:
    - tools are plain Python functions that return a dict
    - Agent(name=..., model=..., instruction=..., tools=[...])
    - the variable must be called root_agent (that is what `adk web` looks for)
    - Runner + InMemorySessionService (in helpdesk/runner.py)
    Different from the template: one agent and no router, Claude, a step limit, steps[] logging.

Run it (from my-work/week-1-rag-api, with the venv active):
    uvicorn main:app --port 8000          then POST /agent (curl examples in README.md)
    python run_helpdesk.py                terminal, prints THINK / ACT / OBSERVE / DONE
    adk web --port 8080                   ADK's browser UI; pick "helpdesk"
"""

import asyncio
import itertools
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException
from google.adk.agents import Agent
from google.adk.models.anthropic_llm import AnthropicLlm

# Load ANTHROPIC_API_KEY and PINECONE_API_KEY from week-1-rag-api/.env (one folder up).
# On Render the keys come from the service's Environment settings instead.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Settings ---

# Which Claude model the agent thinks with. /ask keeps its own model (DEFAULT_MODEL in main.py).
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

# Step limit: the most model calls one message may use. Each THINK is one model call, so 6 is
# plenty for search -> (ticket) -> reply. Set as an environment variable so the limit also
# applies to `adk web`, not only to run_agent().
MAX_STEPS = 6
os.environ.setdefault("ADK_MAX_LLM_CALLS", str(MAX_STEPS))


# --- Tools ---
# The docstring and type hints are what Claude sees, so it knows when and how to call each tool.


async def search_handbook(question: str) -> dict:
    """Search the company documents (handbook, vacation policy, IT policy) for an answer.

    Use this for any question about company rules, time off, benefits, equipment or IT.
    The answer comes only from the documents, with the document_id of each one used.

    Args:
        question: The employee's question, in plain words.

    Returns:
        found: True if the documents answer the question.
        answer: The answer from the documents, or the refusal text.
        citations: document_id of each document used.
        error: Only present if the search itself failed; then nothing else is returned.
    """
    # Imported here, not at the top of the file, because main.py imports this package.
    from main import REFUSAL, AskRequest, ask

    try:
        # ask() is a normal (blocking) function; run it in a thread so the server stays responsive.
        result = await asyncio.to_thread(ask, AskRequest(question=question))
    except HTTPException as exc:  # /ask's own clean failures: 502 (bad answer twice), 503 (Pinecone)
        return {"error": f"The handbook search failed ({exc.status_code}): {exc.detail}"}
    except Exception as exc:  # anything else, e.g. Claude unreachable; report the type, not details
        return {"error": f"The handbook search failed ({type(exc).__name__})."}

    text = result.answer.answer
    citations = result.answer.citations
    found = bool(citations) and REFUSAL not in text
    # Guardrail: if the answer has no citations, don't pass its text on. Otherwise the agent
    # may repeat a half-answer that isn't backed by any document.
    return {
        "found": found,
        "answer": text if found else REFUSAL,
        "citations": citations if found else [],
    }


_ticket_numbers = itertools.count(1)


def create_hr_ticket(summary: str, category: str) -> dict:
    """Open a ticket so a human in HR follows up with the employee.

    Args:
        summary: One sentence describing what the employee needs.
        category: One of "policy_gap" (handbook has no answer) or "sensitive" (needs a human).

    Returns:
        ticket_id and the expected response time.
    """
    # PLACEHOLDER: pretends to create a ticket. A real version would call a ticketing API.
    # The number counts up while the server runs, so each ticket gets its own id.
    response_time = {"sensitive": "4 hours", "policy_gap": "2 business days"}
    return {
        "ticket_id": f"HR-2026-{next(_ticket_numbers):04d}",
        "category": category,
        "summary": summary,
        "expected_response": response_time.get(category, "2 business days"),
    }


# --- The agent ---

root_agent = Agent(
    name="helpdesk_agent",
    model=AnthropicLlm(model=MODEL),
    description="Answers employee policy questions from the handbook, or opens an HR ticket.",
    instruction="""You are the employee help desk.

Goal: help the employee with their question, using only the tools below.

Steps:
1. For any policy question, call search_handbook first.
2. If search_handbook returns found=true, answer using ONLY its answer and list its citations.
3. If found=false, call create_hr_ticket with category "policy_gap" and tell the employee the
   ticket_id and expected_response.
4. If the message is sensitive (harassment, complaints about a person, health, legal),
   do NOT search the handbook: call create_hr_ticket with category "sensitive" right away.
5. If search_handbook returns an "error", tell the employee the handbook search is not
   available right now and to try again later. Do not guess an answer.

Constraints:
- Never invent policy. If the handbook didn't say it, you don't say it. Do not add facts or
  advice from your own knowledge.
- Answer each new question on its own, using ONLY the search_handbook result for that
  question. Do not reuse, compare or correct answers from earlier in the conversation.
- Call each tool at most once per question.
- Before each tool call, write one short sentence saying what you will do and why
  (this is logged as THINK).

Done means: you have sent ONE final reply that either
  (a) answers from the handbook and names the citations, or
  (b) gives the employee a ticket_id and expected response time, or
  (c) says the handbook search is unavailable.""",
    tools=[search_handbook, create_hr_ticket],
)
