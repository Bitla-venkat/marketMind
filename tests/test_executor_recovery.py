"""
Offline fault-injection tests: recovery running inside the executor.

Run from the project root:

    python -m unittest tests.test_executor_recovery -v
"""

import asyncio
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.default_registry import build_default_registry
from core.executor import Executor
from core.models import Lead, LeadList
from core.planner import Planner
from core.recovery import RecoveryPolicy
from core.registry import AgentRegistry, AgentSpec
from core.validation import ValidationResult

from tests.test_pipeline_integration import (
    FakeResponse,
    FakeSearchTool,
    RoutingGemini
)


class ScriptedAgent:
    """
    Behaviour is set by `script(attempt, params, state)`, which may
    raise or set artifacts. `attempt` counts builds of this agent name
    across the whole test (shared via `counter`).
    """

    def __init__(self, name, script, counter, params):

        self.name = name
        self.script = script
        self.counter = counter
        self.params = params
        self.search_tool = SimpleNamespace(search_count=0)

    async def run(self, task, state):

        self.counter[self.name] = self.counter.get(self.name, 0) + 1

        self.search_tool.search_count += 1

        state.llm_calls.append({
            "agent": self.name,
            "model": "m",
            "attempts": 1,
            "latency_s": 0.1,
            "input_tokens": 100,
            "output_tokens": 10,
            "thinking_tokens": 0,
            "failures": []
        })

        self.script(self.counter[self.name], self.params, state)


def make_spec(name, script, counter, requires=(), produces=("leads",),
              **kwargs):

    return AgentSpec(
        name=name,
        description="d",
        requires=requires,
        produces=produces,
        factory=lambda _n=name, **params: ScriptedAgent(
            _n, script, counter, params
        ),
        **kwargs
    )


def set_leads(count):

    def effect(state):
        state.leads = ["lead"] * count

    return effect


def min_leads(n):

    def validator(state):

        found = len(state.leads)

        return ValidationResult(
            found >= n,
            f"found {found}, need {n}",
            {"leads": found}
        )

    return validator


def consumer(counter):
    """A downstream agent that just records it ran."""

    def script(attempt, params, state):
        state.set_artifact("scores", {"x": {"score": 50}})

    return make_spec(
        "consumer",
        script,
        counter,
        requires=("leads",),
        produces=("scores",)
    )


def execute(registry, policy="default", goal="scores", **kwargs):

    if policy == "default":
        policy = RecoveryPolicy(registry)

    plan = Planner(registry).plan(goal)

    return asyncio.run(
        Executor(registry, recovery=policy, **kwargs).run(plan, "s")
    )


class RetryTests(unittest.TestCase):

    def test_transient_failure_is_retried_and_succeeds(self):

        counter = {}

        def flaky(attempt, params, state):

            if attempt == 1:
                raise RuntimeError("503 unavailable")

            state.leads = ["a", "b", "c"]

        registry = AgentRegistry()
        registry.register(make_spec(
            "finder", flaky, counter, escalations=({},)
        ))
        registry.register(consumer(counter))

        state, record = execute(registry)

        self.assertTrue(record["success"])
        self.assertIsNone(record["failed_step"])
        self.assertEqual(
            [(s["agent"], s["attempt"], s["status"])
             for s in record["steps"]],
            [("finder", 1, "failed"),
             ("finder", 2, "ok"),
             ("consumer", 1, "ok")]
        )
        self.assertEqual(len(record["recoveries"]), 1)
        self.assertEqual(record["recoveries"][0]["kind"], "retry")

        # The failed attempt still cost a call and a search.
        self.assertEqual(record["agent_calls"], 3)
        self.assertEqual(record["llm_calls"], 3)
        self.assertEqual(record["search_calls"], 3)

    def test_thin_result_escalates_params_until_it_passes(self):

        counter = {}

        def thin(attempt, params, state):

            state.leads = (
                ["l"] * 5 if params.get("wide") else ["l"]
            )

        registry = AgentRegistry()
        registry.register(make_spec(
            "finder", thin, counter,
            validator=min_leads(3),
            escalations=({}, {"wide": True})
        ))
        registry.register(consumer(counter))

        _, record = execute(registry)

        finder_steps = [
            s for s in record["steps"] if s["agent"] == "finder"
        ]

        self.assertEqual(
            [s["status"] for s in finder_steps],
            ["invalid", "invalid", "ok"]
        )
        self.assertEqual(finder_steps[2]["params"], {"wide": True})
        self.assertTrue(record["success"])
        self.assertEqual(len(record["recoveries"]), 2)

    def test_gives_up_when_nothing_helps(self):

        counter = {}

        registry = AgentRegistry()
        registry.register(make_spec(
            "finder",
            lambda a, p, s: setattr(s, "leads", ["l"]),
            counter,
            validator=min_leads(3),
            escalations=({}, {"wide": True})
        ))
        registry.register(consumer(counter))

        _, record = execute(registry)

        self.assertFalse(record["success"])
        self.assertEqual(record["failed_step"], "finder")
        self.assertEqual(record["skipped"], ["consumer"])
        self.assertEqual(
            [r["kind"] for r in record["recoveries"]],
            ["retry", "retry", "give_up"]
        )

        # 1 first run + 2 retries; the consumer never ran.
        self.assertEqual(counter["finder"], 3)
        self.assertNotIn("consumer", counter)

    def test_no_policy_means_no_recovery(self):

        counter = {}

        registry = AgentRegistry()
        registry.register(make_spec(
            "finder",
            lambda a, p, s: (_ for _ in ()).throw(RuntimeError("x")),
            counter,
            escalations=({},)
        ))
        registry.register(consumer(counter))

        _, record = execute(registry, policy=None)

        self.assertEqual(counter["finder"], 1)
        self.assertEqual(record["recoveries"], [])
        self.assertEqual(record["failed_step"], "finder")


class SwapTests(unittest.TestCase):

    def test_failed_agent_is_swapped_for_an_alternative(self):

        counter = {}

        def broken(attempt, params, state):
            raise RuntimeError("tavily down")

        def backup(attempt, params, state):
            state.leads = ["a", "b", "c"]

        registry = AgentRegistry()
        registry.register(make_spec("primary", broken, counter))
        registry.register(make_spec("backup", backup, counter))
        registry.register(consumer(counter))

        _, record = execute(registry)

        self.assertTrue(record["success"])
        self.assertEqual(
            [(s["node"], s["agent"], s["status"])
             for s in record["steps"]],
            [("primary", "primary", "failed"),
             ("primary", "backup", "ok"),
             ("consumer", "consumer", "ok")]
        )
        self.assertEqual(record["recoveries"][0]["kind"], "swap_agent")
        self.assertEqual(record["recoveries"][0]["agent"], "backup")

    def test_swapped_agent_is_validated_by_its_own_validator(self):

        counter = {}

        registry = AgentRegistry()
        registry.register(make_spec(
            "primary",
            lambda a, p, s: (_ for _ in ()).throw(RuntimeError("x")),
            counter
        ))
        registry.register(make_spec(
            "backup",
            lambda a, p, s: setattr(s, "leads", ["l"]),
            counter,
            validator=min_leads(3)
        ))
        registry.register(consumer(counter))

        _, record = execute(registry)

        self.assertFalse(record["success"])
        self.assertEqual(record["steps"][1]["status"], "invalid")


class BudgetTests(unittest.TestCase):

    def test_run_budget_caps_total_recoveries(self):

        counter = {}

        def always_fails(attempt, params, state):
            raise RuntimeError("x")

        registry = AgentRegistry()
        registry.register(make_spec(
            "finder", always_fails, counter,
            escalations=({}, {}, {})
        ))
        registry.register(consumer(counter))

        policy = RecoveryPolicy(
            registry,
            max_recoveries_per_run=2,
            max_actions_per_node=10
        )

        _, record = execute(registry, policy=policy)

        retries = [
            r for r in record["recoveries"] if r["kind"] == "retry"
        ]

        self.assertEqual(len(retries), 2)
        self.assertEqual(counter["finder"], 3)
        self.assertEqual(record["recoveries"][-1]["kind"], "give_up")
        self.assertIn("budget", record["recoveries"][-1]["reason"])

    def test_recovery_record_is_json_friendly(self):

        import json

        counter = {}

        registry = AgentRegistry()
        registry.register(make_spec(
            "finder",
            lambda a, p, s: (_ for _ in ()).throw(RuntimeError("x")),
            counter,
            escalations=({},)
        ))
        registry.register(consumer(counter))

        _, record = execute(registry)

        json.dumps(record)


class FlakyLeadsGemini(RoutingGemini):
    """Discovery returns `first` leads once, then `later` leads."""

    def __init__(self, first, later):

        super().__init__()

        self.first = first
        self.later = later
        self.discovery_calls = 0

    async def _generate(self, model, contents, config):

        if config["response_schema"] is LeadList:

            self.discovery_calls += 1

            count = (
                self.first if self.discovery_calls == 1 else self.later
            )

            self.schemas.append("LeadList")

            names = ["Alpha", "Beta", "Gamma", "Delta"]

            return FakeResponse(LeadList(leads=[
                Lead(
                    company_name=names[i],
                    reason_for_match="fits",
                    source_url="https://a.example"
                )
                for i in range(count)
            ]))

        return await super()._generate(model, contents, config)


class RealAgentsRecoveryTests(unittest.TestCase):

    def wired(self, gemini, floor=1):

        original = build_default_registry(
            min_leads=3,
            min_leads_floor=floor
        )

        registry = AgentRegistry()

        for spec in original.specs():

            def factory(_spec=spec, **params):

                agent = _spec.factory(**params)

                agent.client = gemini

                if hasattr(agent, "search_tool"):
                    agent.search_tool = FakeSearchTool()

                return agent

            registry.register(AgentSpec(
                name=spec.name,
                description=spec.description,
                requires=spec.requires,
                produces=spec.produces,
                factory=factory,
                default_params=spec.default_params,
                validator=spec.validator,
                escalations=spec.escalations
            ))

        return registry

    def run_default(self, gemini, floor=1):

        registry = self.wired(gemini, floor)

        plan = Planner(registry).plan("scores")

        return asyncio.run(
            Executor(
                registry,
                recovery=RecoveryPolicy(registry)
            ).run(plan, "A billing tool.")
        )

    def test_a_thin_discovery_is_rescued_by_a_plain_retry(self):

        gemini = FlakyLeadsGemini(first=1, later=3)

        _, record = self.run_default(gemini)

        self.assertTrue(record["success"])
        self.assertEqual(gemini.discovery_calls, 2)

        discovery = [
            s for s in record["steps"]
            if s["agent"] == "lead_discovery_agent"
        ]

        self.assertEqual(
            [s["status"] for s in discovery],
            ["invalid", "ok"]
        )
        self.assertEqual(
            [r["kind"] for r in record["recoveries"]],
            ["retry"]
        )

    def test_a_wider_search_is_tried_after_a_plain_retry_fails(self):

        gemini = FlakyLeadsGemini(first=1, later=1)

        # Floor 3 = no tolerance for a thin result.
        _, record = self.run_default(gemini, floor=3)

        # 1 first run, a same-settings retry, then the wide retry.
        self.assertFalse(record["success"])
        self.assertEqual(gemini.discovery_calls, 3)
        self.assertEqual(
            [r["kind"] for r in record["recoveries"]],
            ["retry", "retry", "give_up"]
        )

        wide = record["steps"][-1]

        self.assertEqual(wide["params"]["max_industries"], 3)
        self.assertEqual(
            record["skipped"],
            ["enrichment_agent", "scoring_agent"]
        )

    def test_thin_result_after_all_retries_is_kept_when_acceptable(self):

        gemini = FlakyLeadsGemini(first=1, later=1)

        # Floor 1: one lead is enough to carry on.
        _, record = self.run_default(gemini, floor=1)

        self.assertTrue(record["success"])
        self.assertEqual(gemini.discovery_calls, 3)
        self.assertEqual(record["degraded"], ["lead_discovery_agent"])
        self.assertEqual(record["skipped"], [])
        self.assertEqual(record["failed_step"], None)

    def test_zero_leads_after_all_retries_still_fails(self):

        _, record = self.run_default(
            FlakyLeadsGemini(first=0, later=0),
            floor=1
        )

        self.assertFalse(record["success"])
        self.assertEqual(record["degraded"], [])
        self.assertEqual(record["failed_step"], "lead_discovery_agent")

    def test_a_healthy_run_triggers_no_recovery(self):

        gemini = FlakyLeadsGemini(first=3, later=3)

        _, record = self.run_default(gemini)

        self.assertTrue(record["success"])
        self.assertEqual(record["recoveries"], [])
        self.assertEqual(gemini.discovery_calls, 1)


if __name__ == "__main__":
    unittest.main()
