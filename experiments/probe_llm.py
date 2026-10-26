"""
Make one real ICP call and print the per-call LLM record
(which model answered, and what each failed attempt was).

Run from the project root:

    python -m experiments.probe_llm
"""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from agents.icp_agent import ICPAgent
from core.state import AgentState


async def main():

    state = AgentState(
        "A GST billing tool for small retailers in India."
    )

    await ICPAgent().run({"type": "define_icp"}, state)

    print(json.dumps(state.llm_calls, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())