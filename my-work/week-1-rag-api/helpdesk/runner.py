"""Run the help desk agent and turn ADK's event stream into a final answer plus steps[].

Used by POST /agent and POST /agent/confirm (main.py) and by run_helpdesk.py, so all of
them report the same steps.

How ADK's events map to Think / Act / Observe:
    THINK    text Claude writes before it calls a tool     (text part next to a function_call)
    ACT      the tool Claude chose, with its arguments     (part.function_call)
    OBSERVE  what the tool returned                        (part.function_response)
    answer   Claude's final reply                          (event.is_final_response())

Human-in-the-loop (Session 3 stretch):
    create_hr_ticket is marked require_confirmation=True in agent.py. When the agent wants
    to call it, ADK pauses the run and emits a special "adk_request_confirmation" call
    instead of running the tool. run_agent() returns that as pending_confirmation, and the
    session is kept. resume_agent() sends the human's yes/no to the same session; ADK then
    runs the tool (yes) or tells the agent it was rejected (no), and the agent finishes.

        run_agent("Can I bring my dog?")  -> search_handbook -> wants create_hr_ticket
                                          -> PAUSED, pending_confirmation returned
        resume_agent(..., approved=True)  -> create_hr_ticket runs -> final answer
        resume_agent(..., approved=False) -> no ticket -> agent says so

    Sessions live in this server's memory: a paused run is lost if the server restarts
    (on Render: a redeploy, or the free tier going to sleep). Finished sessions are deleted.
"""

import json

from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import MAX_STEPS, MODEL, root_agent

OBSERVATION_LIMIT = 300  # characters; keeps steps[] short
APP_NAME = "helpdesk"
USER_ID = "employee"
CONFIRMATION_CALL = "adk_request_confirmation"  # ADK's name for "ask the human first"
CONFIRMATION_PENDING_ERROR = "requires confirmation"  # part of the tool's reply while paused

# One session store for the whole server, so a paused run can be resumed by a later request.
_sessions = InMemorySessionService()
_runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=_sessions)
# Paused runs waiting for a human: session_id -> {"confirmation_id", "tool", "args"}
_pending: dict[str, dict] = {}


class UnknownConfirmation(Exception):
    """resume_agent() was given a session/confirmation that isn't waiting for approval."""


def shorten(value, limit: int = OBSERVATION_LIMIT) -> str:
    """One-line JSON, cut off after `limit` characters."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def approval_question(tool: str, args: dict) -> str:
    """The question shown to the human before a consequential tool runs."""
    if tool == "create_hr_ticket":
        return (
            f"I'd like to open an HR ticket ({args.get('category', 'policy_gap')}): "
            f"{args.get('summary', '')} Shall I go ahead?"
        )
    return f"I'd like to run {tool} with {shorten(args)}. Shall I go ahead?"


async def run_agent(message: str) -> dict:
    """Start a new conversation with one message. See _run() for what is returned."""
    session = await _sessions.create_session(app_name=APP_NAME, user_id=USER_ID)
    content = types.Content(role="user", parts=[types.Part(text=message)])
    return await _run(session.id, content)


async def resume_agent(session_id: str, confirmation_id: str, approved: bool) -> dict:
    """Continue a paused run with the human's decision. See _run() for what is returned.

    Raises:
        UnknownConfirmation: nothing with this session_id/confirmation_id is waiting.
    """
    waiting = _pending.get(session_id)
    if not waiting or waiting["confirmation_id"] != confirmation_id:
        raise UnknownConfirmation(session_id)
    del _pending[session_id]
    decision = types.FunctionResponse(
        name=CONFIRMATION_CALL, id=confirmation_id, response={"confirmed": approved}
    )
    content = types.Content(role="user", parts=[types.Part(function_response=decision)])
    return await _run(session_id, content, resumed_call={**waiting, "approved": approved})


async def _run(session_id: str, content: types.Content, resumed_call: dict | None = None) -> dict:
    """Send `content` to the session and collect what happens.

    Returns:
        answer: the final reply ("" if paused or stopped before one).
        steps: one entry per tool call: think, tool, args, observation (shortened).
        model: the model the agent used.
        step_limit_reached: True if ADK stopped the run at MAX_STEPS model calls.
        problem: why there is no answer (None when there is one, or when paused). No secrets.
        pending_confirmation: None, or what the human must approve before the run continues:
            session_id, confirmation_id, tool, args, question.
    """
    steps: list[dict] = []
    answer = ""
    thought: str | None = None
    step_limit_reached = False
    problem: str | None = None  # why there is no answer, if ADK or Claude reported a problem
    finish_reason = None  # why Claude stopped writing its last reply (e.g. STOP, MAX_TOKENS)
    pending: dict | None = None

    try:
        async for event in _runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content,
            run_config=RunConfig(max_llm_calls=MAX_STEPS),
        ):
            if event.error_code:  # ADK reports model failures as events, not exceptions
                problem = f"{event.error_code}: {(event.error_message or '')[:200]}"
            if event.finish_reason:
                finish_reason = event.finish_reason
            parts = event.content.parts if event.content and event.content.parts else []
            for part in parts:
                call, response = part.function_call, part.function_response
                if call and call.name == CONFIRMATION_CALL:  # PAUSE: ask the human first
                    original = (call.args or {}).get("originalFunctionCall", {})
                    pending = {
                        "confirmation_id": call.id,
                        "tool": original.get("name", "?"),
                        "args": dict(original.get("args") or {}),
                    }
                elif call:  # ACT
                    steps.append({
                        "think": thought,
                        "tool": call.name,
                        "args": dict(call.args or {}),
                        "observation": None,
                    })
                    thought = None
                elif response:  # OBSERVE: attach to the matching ACT
                    observation = shorten(response.response)
                    if CONFIRMATION_PENDING_ERROR in observation:
                        observation = "PAUSED: waiting for human approval"
                    step = next(
                        (s for s in reversed(steps)
                         if s["tool"] == response.name and s["observation"] is None),
                        None,
                    )
                    if step is None and resumed_call and response.name == resumed_call["tool"]:
                        # After the human's decision the paused tool runs (or is skipped) now;
                        # its ACT was in the first run, so record it here with the decision.
                        decision = "Approved" if resumed_call["approved"] else "Rejected"
                        step = {"think": f"{decision} by a human.", "tool": response.name,
                                "args": resumed_call["args"], "observation": None}
                        steps.append(step)
                    if step is not None:
                        step["observation"] = observation
                elif part.text and event.is_final_response():  # final answer
                    answer += part.text
                elif part.text:  # THINK
                    thought = f"{thought} {part.text.strip()}" if thought else part.text.strip()
    except LlmCallsLimitExceededError:
        step_limit_reached = True

    answer = answer.strip()
    pending_confirmation = None
    if pending:
        _pending[session_id] = pending
        pending_confirmation = {
            "session_id": session_id,
            **pending,
            "question": approval_question(pending["tool"], pending["args"]),
        }
        answer = pending_confirmation["question"]
    else:
        await _sessions.delete_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    if not answer and not step_limit_reached and problem is None:
        problem = f"Claude ended without a text reply (finish reason: {finish_reason})."
    if not answer:
        # Printed to the server log (Render: Logs tab), so failures can be diagnosed later.
        print(f"[run_agent] no answer: problem={problem!r} steps={[s['tool'] for s in steps]}", flush=True)

    return {
        "answer": answer,
        "steps": steps,
        "model": MODEL,
        "step_limit_reached": step_limit_reached,
        "problem": None if answer else problem,
        "pending_confirmation": pending_confirmation,
    }
