"""
Offline tests for core/executor.py.

Run from the project root:

    python -m unittest tests.test_executor -v
"""

import asyncio
import os
import time
import unittest
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.default_registry import build_default_registry
from core.executor import Executor
from core.planner import Planner
from core.registry import AgentRegistry, AgentSpec
from core.models import EnrichedLead, Lead
from pipelines.fixed_pipeline import FixedPipeline

from tests.test_pipeline_integration import (
    FakeSearchTool,
    RoutingGemini
)


class FakeAgent:
    """Records that it ran, optionally sleeps, raises or costs."""

    def __init__(
        self,
        name="a",
        delay=0.0,
        fail=False,
        llm_calls=0,
        searches=0,
        log=None,
        **_ignored
    ):
        self.name = name
        self.delay = delay
        self.fail = fail
        self.llm_calls = llm_calls
        self.searches = searches
        self.log = log if log is not None else []
        self.search_tool = SimpleNamespace(search_count=0)

    async def run(self, task, state):

        self.log.append(("start", self.name))

        await asyncio.sleep(self.delay)

        self.search_tool.search_count += self.searches

        for _ in range(self.llm_calls):
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

        if self.fail:
            raise RuntimeError(f"{self.name} broke")

        state.set_artifact(f"out_{self.name}", ["x"])

        self.log.append(("end", self.name))


def registry_of(*items):
    """items: (name, requires, produces, factory_kwargs)."""

    registry = AgentRegistry()

    for name, requires, produces, kwargs in items:

        registry.register(AgentSpec(
            name=name,
            description="d",
            requires=requires,
            produces=produces,
            factory=lambda _name=name, _kw=kwargs, **params: (
                FakeAgent(name=_name, **{**_kw, **params})
            )
        ))

    return registry


def run(registry, goal, **executor_kwargs):

    plan = Planner(registry).plan(goal)

    return asyncio.run(
        Executor(registry, **executor_kwargs).run(plan, "a startup")
    )


class ExecutionOrderTests(unittest.TestCase):

    def test_dependencies_run_before_dependents(self):

        log = []

        registry = registry_of(
            ("a", (), ("x",), {"log": log}),
            ("b", ("x",), ("goal",), {"log": log})
        )

        _, record = run(registry, "goal")

        self.assertEqual(
            log,
            [("start", "a"), ("end", "a"),
             ("start", "b"), ("end", "b")]
        )
        self.assertEqual(
            [s["agent"] for s in record["steps"]],
            ["a", "b"]
        )

    def test_independent_nodes_run_in_parallel(self):

        registry = registry_of(
            ("a", (), ("x",), {}),
            ("b", ("x",), ("y",), {"delay": 0.2}),
            ("c", ("x",), ("z",), {"delay": 0.2}),
            ("d", ("y", "z"), ("goal",), {})
        )

        started = time.perf_counter()

        _, record = run(registry, "goal")

        elapsed = time.perf_counter() - started

        self.assertEqual(
            [s["status"] for s in record["steps"]],
            ["ok"] * 4
        )

        # Two 0.2s nodes in parallel: well under the sequential 0.4s.
        self.assertLess(elapsed, 0.35)

    def test_max_concurrency_one_runs_them_one_at_a_time(self):

        registry = registry_of(
            ("a", (), ("x",), {}),
            ("b", ("x",), ("y",), {"delay": 0.15}),
            ("c", ("x",), ("z",), {"delay": 0.15}),
            ("d", ("y", "z"), ("goal",), {})
        )

        started = time.perf_counter()

        run(registry, "goal", max_concurrency=1)

        self.assertGreaterEqual(time.perf_counter() - started, 0.3)

    def test_invalid_concurrency_rejected(self):

        with self.assertRaises(ValueError):
            Executor(AgentRegistry(), max_concurrency=0)

    def test_params_reach_the_agent(self):

        log = []

        registry = AgentRegistry()
        built = []

        def factory(**params):
            built.append(params)
            return FakeAgent(name="a", log=log)

        registry.register(AgentSpec(
            name="a",
            description="d",
            produces=("goal",),
            factory=factory,
            default_params={"n": 1}
        ))

        plan = Planner(registry).plan("goal", params={"a": {"n": 7}})

        asyncio.run(Executor(registry).run(plan, "s"))

        self.assertEqual(built, [{"n": 7}])


class FailureTests(unittest.TestCase):

    def test_failure_skips_dependents_but_not_independent_nodes(self):

        registry = registry_of(
            ("a", (), ("x",), {}),
            ("bad", ("x",), ("y",), {"fail": True}),
            ("ok", ("x",), ("z",), {}),
            ("d", ("y", "z"), ("goal",), {})
        )

        state, record = run(registry, "goal")

        by_agent = {s["agent"]: s for s in record["steps"]}

        self.assertEqual(by_agent["bad"]["status"], "failed")
        self.assertEqual(by_agent["ok"]["status"], "ok")
        self.assertEqual(record["failed_step"], "bad")
        self.assertIn("bad broke", record["error"])
        self.assertEqual(record["skipped"], ["d"])
        self.assertFalse(record["success"])
        self.assertTrue(state.has_artifact("out_ok"))

    def test_build_errors_count_as_node_failures(self):

        registry = AgentRegistry()

        def broken(**_params):
            raise ValueError("GEMINI_API_KEY is not set.")

        registry.register(AgentSpec(
            name="a",
            description="d",
            produces=("goal",),
            factory=broken
        ))

        _, record = run(registry, "goal")

        self.assertEqual(record["failed_step"], "a")
        self.assertEqual(record["search_calls"], 0)
        self.assertEqual(record["steps"][0]["status"], "failed")

    def test_failed_llm_attempts_on_exception_are_counted(self):

        class Failing(FakeAgent):

            async def run(self, task, state):

                error = RuntimeError("down")
                error.llm_failures = [{"code": 503}, {"code": 503}]

                raise error

        registry = AgentRegistry()
        registry.register(AgentSpec(
            name="a",
            description="d",
            produces=("goal",),
            factory=lambda **p: Failing(name="a")
        ))

        _, record = run(registry, "goal")

        self.assertEqual(record["llm_failed_attempts"], 2)


class AccountingTests(unittest.TestCase):

    def test_parallel_nodes_are_costed_separately(self):

        registry = registry_of(
            ("a", (), ("x",), {}),
            ("b", ("x",), ("y",), {
                "llm_calls": 2, "searches": 3, "delay": 0.05
            }),
            ("c", ("x",), ("z",), {
                "llm_calls": 1, "searches": 1, "delay": 0.05
            }),
            ("d", ("y", "z"), ("goal",), {})
        )

        _, record = run(registry, "goal")

        by_agent = {s["agent"]: s for s in record["steps"]}

        self.assertEqual(by_agent["b"]["llm_calls"], 2)
        self.assertEqual(by_agent["b"]["search_calls"], 3)
        self.assertEqual(by_agent["c"]["llm_calls"], 1)
        self.assertEqual(by_agent["c"]["search_calls"], 1)
        self.assertEqual(record["llm_calls"], 3)
        self.assertEqual(record["search_calls"], 4)
        self.assertEqual(record["input_tokens"], 300)
        self.assertEqual(record["agent_calls"], 4)

    def test_record_has_the_same_fields_as_the_fixed_pipeline(self):

        registry = registry_of(("a", (), ("goal",), {}))

        _, dynamic = run(registry, "goal")

        _, fixed = asyncio.run(
            FixedPipeline("f", [(FakeAgent("a"), None)]).run("x")
        )

        self.assertEqual(
            set(fixed) - set(dynamic),
            set()
        )
        self.assertEqual(
            set(dynamic) - set(fixed),
            {"skipped", "recoveries", "plan", "max_concurrency"}
        )


class RealAgentsOfflineTests(unittest.TestCase):
    """The four real agents, planned and executed, with fake clients."""

    def wired_registry(self, gemini):

        original = build_default_registry()

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
                default_params=spec.default_params
            ))

        return registry

    def test_matches_the_fixed_pipeline_numbers(self):

        gemini = RoutingGemini()

        registry = self.wired_registry(gemini)

        plan = Planner(registry).plan("scores")

        state, record = asyncio.run(
            Executor(registry).run(plan, "A billing tool.", label="gst")
        )

        # Same expectations as the fixed pipeline's integration test.
        self.assertTrue(record["success"])
        self.assertEqual(record["pipeline"], "dynamic")
        self.assertEqual(record["agent_calls"], 4)
        self.assertEqual(record["llm_calls"], 5)
        self.assertEqual(record["search_calls"], 5)
        self.assertEqual(record["leads_found"], 2)
        self.assertEqual(record["leads_enriched_ok"], 2)
        self.assertEqual(record["leads_scored"], 2)
        self.assertEqual(record["input_tokens"], 500)
        self.assertEqual(list(state.scores), ["Alpha", "Beta"])
        self.assertEqual(record["skipped"], [])


if __name__ == "__main__":
    unittest.main()
