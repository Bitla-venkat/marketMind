"""
Frozen ICPs: one fixed ICP per example startup.

The ICP agent is non-deterministic, so if every pipeline run asks it
for a fresh ICP, runs differ because of the ICP and not because of the
pipeline. This module generates an ICP once per startup, saves it to
experiments/frozen_icps.json, and reuses it on every later run.

An entry is regenerated only if the startup's description changed.
Delete the file to start over. Commit it: it is a benchmark input.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from core.models import ICP
from core.state import AgentState


FROZEN_PATH = Path(__file__).parent / "frozen_icps.json"


async def generate_icp(description: str) -> ICP:
    """One real ICP agent call."""

    from agents.icp_agent import ICPAgent

    state = AgentState(description)

    await ICPAgent().run({"type": "define_icp"}, state)

    return state.icp


def _load(path: Path) -> dict:

    if not path.exists():
        return {}

    try:

        data = json.loads(path.read_text())

    except ValueError:

        print(f"{path.name} is not valid JSON; regenerating it.")

        return {}

    return data if isinstance(data, dict) else {}


async def get_frozen_icps(
    startups: list[dict],
    path: Path = FROZEN_PATH,
    generate=generate_icp
) -> dict:
    """
    {startup id: ICP} for the given startups, generating and saving
    any that are missing or whose description changed.

    `generate` is an async function description -> ICP. It is a
    parameter so tests can run without an API.
    """

    stored = _load(path)

    icps = {}

    changed = False

    for startup in startups:

        entry = stored.get(startup["id"])

        if entry and entry.get("description") == startup["description"]:

            try:

                icps[startup["id"]] = ICP.model_validate(entry["icp"])

                continue

            except ValueError:

                print(
                    f"Saved ICP for {startup['id']} is invalid; "
                    "regenerating it."
                )

        print(f"Generating a frozen ICP for {startup['id']}...")

        icp = await generate(startup["description"])

        stored[startup["id"]] = {
            "description": startup["description"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "icp": icp.model_dump()
        }

        icps[startup["id"]] = icp

        changed = True

    if changed:

        path.write_text(json.dumps(stored, indent=2))

        print(f"Saved frozen ICPs to {path}")

    return icps
