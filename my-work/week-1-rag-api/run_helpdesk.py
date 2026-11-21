"""Run the help desk agent in the terminal and print each step as THINK / ACT / OBSERVE / DONE.

Run (from my-work/week-1-rag-api, with the venv active). No server needed: the agent calls
ask() from main.py directly.
    python run_helpdesk.py                                        # the three test questions
    python run_helpdesk.py "How many vacation days carry over?"   # your own question
    python run_helpdesk.py --yes "Can I bring my dog?"            # auto-approve tickets
    python run_helpdesk.py --no  "Can I bring my dog?"            # auto-reject tickets

Human-in-the-loop: before create_hr_ticket the agent pauses (WAITING) and you are asked
"Approve? [y/n]" — unless --yes / --no answers for you.

Uses the same run_agent() / resume_agent() as POST /agent and /agent/confirm, so the
steps here match the API's steps[].
"""

import asyncio
import sys

from helpdesk.agent import MAX_STEPS, MODEL
from helpdesk.runner import resume_agent, run_agent

TESTS = [
    ("IN THE HANDBOOK", "Do my unused vacation days carry over to next year?"),
    ("NOT IN THE HANDBOOK", "Can I bring my dog to the office?"),
    ("SENSITIVE", "My manager keeps making comments about my health. What can I do?"),
]


def print_steps(result: dict, first_number: int) -> int:
    """Print THINK / ACT / OBSERVE for each step; return the next step number."""
    number = first_number
    for step in result["steps"]:
        if step["think"]:
            print(f"THINK    {step['think']}")
        print(f"ACT      [step {number}] {step['tool']}({step['args']})")
        print(f"OBSERVE  {step['observation']}")
        number += 1
    return number


def decide(auto: bool | None) -> bool:
    """The human's decision: from --yes/--no, or typed in."""
    if auto is not None:
        print(f"HUMAN    {'yes' if auto else 'no'} (from --{'yes' if auto else 'no'})")
        return auto
    return input("HUMAN    Approve? [y/n] ").strip().lower().startswith("y")


async def run(question: str, auto: bool | None) -> None:
    print(f"USER     {question}")
    result = await run_agent(question)
    number = print_steps(result, 1)

    while result["pending_confirmation"]:  # paused before a consequential tool
        pending = result["pending_confirmation"]
        print(f"WAITING  {pending['question']}")
        approved = decide(auto)
        result = await resume_agent(pending["session_id"], pending["confirmation_id"], approved)
        number = print_steps(result, number)

    if result["step_limit_reached"]:
        print(f"STOPPED  step limit of {MAX_STEPS} model calls reached - no final answer.")
    else:
        print(f"DONE     {result['answer']}")


async def main() -> None:
    args = sys.argv[1:]
    auto = None  # None = ask the human in the terminal
    if "--yes" in args:
        auto = True
    elif "--no" in args:
        auto = False
    args = [a for a in args if a not in ("--yes", "--no")]

    print(f"Model: {MODEL}   Step limit: {MAX_STEPS} model calls per question")
    if args:
        await run(" ".join(args), auto)
        return
    for label, question in TESTS:
        print(f"\n--- {label} ---")
        await run(question, auto)


if __name__ == "__main__":
    asyncio.run(main())
