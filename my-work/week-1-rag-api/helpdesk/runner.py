"""Run the help desk agent once and turn ADK's event stream into a final answer plus steps[].

Used by POST /agent (main.py) and by run_helpdesk.py, so both report the same steps.

How ADK's events map to Think / Act / Observe:
    THINK    text Claude writes before it calls a tool     (text part next to a function_call)
    ACT      the tool Claude chose, with its arguments     (part.function_call)
    OBSERVE  what the tool returned                        (part.function_response)
    answer   Claude's final reply                          (event.is_final_response())

Each call gets a fresh session, so every message is handled on its own (like /ask).
"""

import json

from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import MAX_STEPS, MODEL, root_agent

OBSERVATION_LIMIT = 300  # characters; keeps steps[] short


def shorten(value, limit: int = OBSERVATION_LIMIT) -> str:
    """One-line JSON, cut off after `limit` characters."""
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


async def run_agent(message: str) -> dict:
    """Run one message through the agent.

    Returns:
        answer: the final reply ("" if the step limit stopped the run first).
        steps: one entry per tool call: think, tool, args, observation (shortened).
        model: the model the agent used.
        step_limit_reached: True if ADK stopped the run at MAX_STEPS model calls.
        problem: why there is no answer (None when there is one). Short, no secrets.
    """
    service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="helpdesk", session_service=service)
    session = await service.create_session(app_name="helpdesk", user_id="employee")
    content = types.Content(role="user", parts=[types.Part(text=message)])

    steps: list[dict] = []
    answer = ""
    thought: str | None = None
    step_limit_reached = False
    problem: str | None = None  # why there is no answer, if ADK or Claude reported a problem
    finish_reason = None  # why Claude stopped writing its last reply (e.g. STOP, MAX_TOKENS)

    try:
        async for event in runner.run_async(
            user_id="employee",
            session_id=session.id,
            new_message=content,
            run_config=RunConfig(max_llm_calls=MAX_STEPS),
        ):
            if event.error_code:  # ADK reports model failures as events, not exceptions
                problem = f"{event.error_code}: {(event.error_message or '')[:200]}"
            if event.finish_reason:
                finish_reason = event.finish_reason
            parts = event.content.parts if event.content and event.content.parts else []
            for part in parts:
                if part.function_call:  # ACT
                    steps.append({
                        "think": thought,
                        "tool": part.function_call.name,
                        "args": dict(part.function_call.args or {}),
                        "observation": None,
                    })
                    thought = None
                elif part.function_response:  # OBSERVE: attach to the matching ACT
                    for step in reversed(steps):
                        if step["tool"] == part.function_response.name and step["observation"] is None:
                            step["observation"] = shorten(part.function_response.response)
                            break
                elif part.text and event.is_final_response():  # final answer
                    answer += part.text
                elif part.text:  # THINK
                    thought = f"{thought} {part.text.strip()}" if thought else part.text.strip()
    except LlmCallsLimitExceededError:
        step_limit_reached = True

    answer = answer.strip()
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
    }
