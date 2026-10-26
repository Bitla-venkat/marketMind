"""
The fixed sequential baseline pipeline.

Runs a hard-wired list of agents one after another, with NO
validation, recovery or replanning:

    ICP -> Lead discovery -> Enrichment -> Scoring

If a step raises, the pipeline stops there and the run is marked
failed. A step that finishes but produces nothing (for example zero
leads) is not treated as a failure at the time; the next step simply
runs, and normally fails because its input is missing. That is
deliberate: this is the reference the dynamic orchestrator is measured
against.

Every run returns a plain-dict run record. The dynamic orchestrator
should produce records with the same fields so the two can be compared
directly.

A run counts as a SUCCESS when every step finished without raising AND
at least one lead was scored. Change `is_success` if you want a
different definition.
"""

import time
from datetime import datetime, timezone

from agents.enrichment_agent import EnrichmentAgent
from agents.icp_agent import ICPAgent
from agents.lead_discovery_agent import LeadDiscoveryAgent
from agents.scoring_agent import ScoringAgent
from core.state import AgentState


# Discovery settings for the two fixed variants. They match the
# "baseline" and "wide" settings used in experiments/probe_discovery.py.
VARIANTS = {
    "narrow": {
        "max_industries": 1,
        "results_per_query": 3
    },
    "wide": {
        "max_industries": 3,
        "results_per_query": 5
    }
}


def is_success(record: dict) -> bool:
    """Every step finished AND at least one lead was scored."""

    return (
        record["failed_step"] is None
        and record["leads_scored"] >= 1
    )


def _search_count(agent) -> int:
    """Searches attempted so far by an agent's search tool."""

    return getattr(
        getattr(agent, "search_tool", None),
        "search_count",
        0
    )


def _failed_attempts(calls: list[dict]) -> int:
    """Gemini attempts that failed inside calls that recovered."""

    return sum(len(call["failures"]) for call in calls)


class FixedPipeline:

    def __init__(self, name: str, steps: list):
        """
        name:
            Label stored in every run record, e.g. "fixed-narrow".

        steps:
            List of (agent, task) pairs, run in order.
        """

        self.name = name
        self.steps = steps

    async def run(
        self,
        startup_description: str,
        label: str | None = None
    ):
        """
        Run every step in order.

        Returns (state, record). `label` identifies the input
        (for example a startup id) in the record.
        """

        state = AgentState(startup_description)

        record = {
            "pipeline": self.name,
            "label": label,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "failed_step": None,
            "error": None,
            "latency_s": None,
            "agent_calls": 0,
            "llm_calls": 0,
            "llm_failed_attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "search_calls": 0,
            "leads_found": 0,
            "leads_enriched_ok": 0,
            "leads_scored": 0,
            "steps": []
        }

        started = time.perf_counter()

        for agent, task in self.steps:

            step = await self._run_step(agent, task, state)

            record["steps"].append(step)

            if step["status"] != "ok":

                record["failed_step"] = step["agent"]
                record["error"] = step["error"]

                break

        record["latency_s"] = round(
            time.perf_counter() - started,
            3
        )

        # Totals come from the steps, so failed steps count too.
        for step in record["steps"]:

            record["agent_calls"] += 1
            record["llm_calls"] += step["llm_calls"]
            record["llm_failed_attempts"] += step[
                "llm_failed_attempts"
            ]
            record["input_tokens"] += step["input_tokens"]
            record["output_tokens"] += step["output_tokens"]
            record["search_calls"] += step["search_calls"]

        record["leads_found"] = len(state.leads)

        record["leads_enriched_ok"] = sum(
            1
            for enriched in state.enriched_leads
            if enriched.status == "ok"
        )

        record["leads_scored"] = len(state.scores)

        record["success"] = is_success(record)

        return state, record

    async def _run_step(self, agent, task, state) -> dict:
        """Run one agent and measure what it cost."""

        llm_before = len(state.llm_calls)
        searches_before = _search_count(agent)

        status = "ok"
        error = None
        extra_failed_attempts = 0

        started = time.perf_counter()

        try:

            await agent.run(task=task, state=state)

        except Exception as e:

            status = "failed"

            error = f"{type(e).__name__}: {str(e)[:300]}"

            # Gemini attempts made before a call failed outright.
            extra_failed_attempts = len(
                getattr(e, "llm_failures", [])
            )

            print(f"\nStep {agent.name} failed: {error}")

        latency = time.perf_counter() - started

        calls = state.llm_calls[llm_before:]

        return {
            "agent": agent.name,
            "status": status,
            "error": error,
            "latency_s": round(latency, 3),
            "llm_calls": len(calls),
            "llm_failed_attempts": (
                _failed_attempts(calls) + extra_failed_attempts
            ),
            "input_tokens": sum(
                call["input_tokens"] or 0 for call in calls
            ),
            "output_tokens": sum(
                call["output_tokens"] or 0 for call in calls
            ),
            "search_calls": (
                _search_count(agent) - searches_before
            )
        }


def build_fixed_pipeline(variant: str = "narrow") -> FixedPipeline:
    """
    Build a fixed pipeline with real agents.

    variant:
        "narrow" : discovery searches 1 industry, 3 results/query
        "wide"   : discovery searches 3 industries, 5 results/query
    """

    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from: {', '.join(VARIANTS)}."
        )

    return FixedPipeline(
        name=f"fixed-{variant}",
        steps=[
            (ICPAgent(), {"type": "define_icp"}),
            (
                LeadDiscoveryAgent(**VARIANTS[variant]),
                {"type": "discover_leads"}
            ),
            (EnrichmentAgent(), None),
            (ScoringAgent(), {"type": "score_leads"})
        ]
    )
