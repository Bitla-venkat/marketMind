"""
Executor: runs a Plan against an AgentState.

Starts every node whose dependencies are finished, in parallel (up to
`max_concurrency` at a time), and records what each node cost.

Failure handling:

  * A node that raises is "failed"; a node whose output the agent's
    validator rejects is "invalid".
  * With no recovery policy, either one is final: nodes that depend
    on it are skipped and independent nodes still run.
  * With a recovery policy (core/recovery.py), the policy first picks
    an action: retry (possibly with changed params), swap in another
    agent, or give up. Every attempt is a step in the run record and
    costs what it costs; each action is logged under "recoveries".

The run record has the same fields as FixedPipeline's, plus the plan,
so the two can be compared directly.
"""

import asyncio
import time
from dataclasses import replace
from datetime import datetime, timezone

from core.planner import Plan
from core.registry import AgentRegistry
from core.state import AgentState
from pipelines.fixed_pipeline import is_success, models_used


def _search_count(agent) -> int:
    """Searches attempted so far by an agent's search tool."""

    return getattr(
        getattr(agent, "search_tool", None),
        "search_count",
        0
    )


class Executor:

    def __init__(
        self,
        registry: AgentRegistry,
        max_concurrency: int = 4,
        name: str = "dynamic",
        validate: bool = True,
        recovery=None
    ):

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")

        self.registry = registry
        self.max_concurrency = max_concurrency
        self.name = name

        # Run each agent's validator (if it has one) after the agent
        # finishes. A node whose output is rejected gets status
        # "invalid": its dependents are skipped, like a failed node.
        self.validate = validate

        # RecoveryPolicy or None. None means failures are final.
        self.recovery = recovery

    async def run(
        self,
        plan: Plan,
        startup_description: str,
        label: str | None = None
    ):
        """
        Execute the plan from scratch. Returns (state, record).
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
            "models_used": {},
            "leads_found": 0,
            "leads_enriched_ok": 0,
            "leads_scored": 0,
            "steps": [],
            "skipped": [],
            "recoveries": [],
            "plan": plan.to_dict(),
            "max_concurrency": self.max_concurrency
        }

        # Nodes may be replaced during recovery, so work on a copy.
        work = Plan(goal=plan.goal, nodes=list(plan.nodes))

        # node id -> recovery actions taken for it so far.
        taken = {}

        budget_left = (
            self.recovery.max_recoveries_per_run
            if self.recovery is not None
            else 0
        )

        # node id -> step record of its final failed attempt.
        final_failures = {}

        semaphore = asyncio.Semaphore(self.max_concurrency)

        done = set()
        failed = set()

        started = time.perf_counter()

        while True:

            ready = [
                node
                for node in work.ready(done)
                if node.id not in failed
            ]

            if not ready:
                break

            steps = await asyncio.gather(*[
                self._run_node(
                    node,
                    work,
                    state,
                    semaphore,
                    attempt=len(taken.get(node.id, [])) + 1
                )
                for node in ready
            ])

            for node, step in zip(ready, steps):

                record["steps"].append(step)

                if step["status"] == "ok":
                    done.add(node.id)
                    continue

                prior = taken.setdefault(node.id, [])

                if self.recovery is None:
                    action = None

                else:
                    action = self.recovery.decide(
                        node,
                        step,
                        prior,
                        budget_left
                    )

                if action is None or action.kind == "give_up":

                    failed.add(node.id)
                    final_failures[node.id] = step

                    if action is not None:
                        record["recoveries"].append({
                            "node": node.id,
                            **action.to_dict()
                        })

                    continue

                budget_left -= 1

                prior.append(action)

                record["recoveries"].append({
                    "node": node.id,
                    **action.to_dict()
                })

                work.nodes[work.nodes.index(node)] = (
                    self._apply(node, action)
                )

        record["latency_s"] = round(
            time.perf_counter() - started,
            3
        )

        record["skipped"] = [
            node.id
            for node in plan.nodes
            if node.id not in done and node.id not in failed
        ]

        # The first node (in plan order) that ended in failure.
        for node in plan.nodes:

            if node.id in final_failures:

                record["failed_step"] = node.id
                record["error"] = final_failures[node.id]["error"]

                break

        for step in record["steps"]:

            record["agent_calls"] += 1
            record["llm_calls"] += step["llm_calls"]
            record["llm_failed_attempts"] += step[
                "llm_failed_attempts"
            ]
            record["input_tokens"] += step["input_tokens"]
            record["output_tokens"] += step["output_tokens"]
            record["search_calls"] += step["search_calls"]

        record["models_used"] = models_used(state.llm_calls)

        record["leads_found"] = len(state.leads)

        record["leads_enriched_ok"] = sum(
            1
            for enriched in state.enriched_leads
            if enriched.status == "ok"
        )

        record["leads_scored"] = len(state.scores)

        record["success"] = is_success(record)

        return state, record

    def _apply(self, node, action):
        """The node to run next, after a recovery action."""

        if action.kind == "retry":

            return replace(
                node,
                params={**node.params, **action.params}
            )

        # swap_agent: same place in the graph, different agent.
        spec = self.registry.get(action.agent)

        return replace(
            node,
            agent=spec.name,
            params=dict(spec.default_params),
            requires=tuple(spec.requires),
            produces=tuple(spec.produces),
            alternatives=tuple(
                name
                for name in (*node.alternatives, node.agent)
                if name != spec.name
            )
        )

    async def _run_node(
        self,
        node,
        plan,
        state,
        semaphore,
        attempt: int = 1
    ) -> dict:
        """Build and run one agent, measuring what it cost."""

        async with semaphore:

            status = "ok"
            error = None
            validation = None
            extra_failed_attempts = 0

            # LLM calls are matched to a node by agent name, because
            # several nodes may run at once and share state.llm_calls.
            calls_before = self._calls_for(state, node.agent)

            agent = None
            searches_before = 0

            started = time.perf_counter()

            try:

                agent = self.registry.build(
                    node.agent,
                    **node.params
                )

                searches_before = _search_count(agent)

                await agent.run(
                    task={"goal": plan.goal, "node": node.id},
                    state=state
                )

                if self.validate:

                    validation = self._validate(node, state)

                    if (
                        validation is not None
                        and not validation["passed"]
                    ):
                        status = "invalid"
                        error = validation["reason"]

                        print(
                            f"\nStep {node.id} output rejected: "
                            f"{error}"
                        )

            except Exception as e:

                status = "failed"

                error = f"{type(e).__name__}: {str(e)[:300]}"

                extra_failed_attempts = len(
                    getattr(e, "llm_failures", [])
                )

                print(f"\nStep {node.id} failed: {error}")

            latency = time.perf_counter() - started

            calls = self._calls_for(state, node.agent)[
                len(calls_before):
            ]

            return {
                "node": node.id,
                "agent": node.agent,
                "attempt": attempt,
                "params": dict(node.params),
                "status": status,
                "error": error,
                "validation": validation,
                "latency_s": round(latency, 3),
                "llm_calls": len(calls),
                "llm_failed_attempts": (
                    sum(len(call["failures"]) for call in calls)
                    + extra_failed_attempts
                ),
                "input_tokens": sum(
                    call["input_tokens"] or 0 for call in calls
                ),
                "output_tokens": sum(
                    call["output_tokens"] or 0 for call in calls
                ),
                "search_calls": (
                    _search_count(agent) - searches_before
                    if agent is not None
                    else 0
                )
            }

    def _validate(self, node, state) -> dict | None:
        """Run the agent's validator. None if it has none."""

        validator = self.registry.get(node.agent).validator

        if validator is None:
            return None

        try:

            return validator(state).to_dict()

        except Exception as e:

            return {
                "passed": False,
                "reason": (
                    f"Validator error: {type(e).__name__}: "
                    f"{str(e)[:200]}"
                ),
                "metrics": {}
            }

    @staticmethod
    def _calls_for(state, agent_name: str) -> list:

        return [
            call
            for call in state.llm_calls
            if call["agent"] == agent_name
        ]
