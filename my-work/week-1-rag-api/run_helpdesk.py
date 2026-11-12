"""Run the help desk agent in the terminal and print each step as THINK / ACT / OBSERVE / DONE.

Run (from my-work/week-1-rag-api, with the venv active). No server needed: the agent calls
ask() from main.py directly.
    python run_helpdesk.py                                        # the three test questions
    python run_helpdesk.py "How many vacation days carry over?"   # your own question

Uses the same run_agent() as POST /agent, so the steps here match the API's steps[].
"""

import asyncio
import sys

from helpdesk.agent import MAX_STEPS, MODEL
from helpdesk.runner import run_agent

TESTS = [
    ("IN THE HANDBOOK", "Do my unused vacation days carry over to next year?"),
    ("NOT IN THE HANDBOOK", "Can I bring my dog to the office?"),
    ("SENSITIVE", "My manager keeps making comments about my health. What can I do?"),
]


async def run(question: str) -> None:
    print(f"USER     {question}")
    result = await run_agent(question)
    for number, step in enumerate(result["steps"], start=1):
        if step["think"]:
            print(f"THINK    {step['think']}")
        print(f"ACT      [step {number}] {step['tool']}({step['args']})")
        print(f"OBSERVE  {step['observation']}")
    if result["step_limit_reached"]:
        print(f"STOPPED  step limit of {MAX_STEPS} model calls reached - no final answer.")
    else:
        print(f"DONE     {result['answer']}")


async def main() -> None:
    print(f"Model: {MODEL}   Step limit: {MAX_STEPS} model calls per question")
    if len(sys.argv) > 1:
        await run(" ".join(sys.argv[1:]))
        return
    for label, question in TESTS:
        print(f"\n--- {label} ---")
        await run(question)


if __name__ == "__main__":
    asyncio.run(main())
