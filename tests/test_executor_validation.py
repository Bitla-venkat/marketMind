"""
Offline tests: validators running inside the executor.

Run from the project root:

    python -m unittest tests.test_executor_validation -v
"""

import asyncio
import os
import unittest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.default_registry import build_default_registry
from core.executor import Executor
from core.planner import Planner
from core.registry import AgentRegistry, AgentSpec
from core.validation import ValidationResult

from tests.test_executor import FakeAgent
from tests.test_pipeline_integration import (
    FakeSearchTool,
    RoutingGemini
)


def make_registry(validator_b):

    registry = AgentRegistry()

    registry.register(AgentSpec(
        name="a",
        description="d",
        produces=("x",),
        factory=lambda **p: FakeAgent(name="a")
    ))

    registry.register(AgentSpec(
        name="b",
        description="d",
        requires=("x",),
        produces=("goal",),
        factory=lambda **p: FakeAgent(name="b"),
        validator=validator_b
    ))

    return registry


def run(registry, **kwargs):

    plan = Planner(registry).plan("goal")

    return asyncio.run(
        Executor(registry, **kwargs).run(plan, "a startup")
    )


class ExecutorValidationTests(unittest.TestCase):

    def test_passing_validator_is_recorded(self):

        registry = make_registry(
            lambda state: ValidationResult(True, "", {"n": 1})
        )

        _, record = run(registry)

        step = record["steps"][1]

        self.assertEqual(step["status"], "ok")
        self.assertEqual(
            step["validation"],
            {"passed": True, "reason": "", "metrics": {"n": 1}}
        )

    def test_agent_without_validator_has_none(self):

        _, record = run(make_registry(None))

        self.assertIsNone(record["steps"][0]["validation"])

    def test_rejected_output_marks_the_node_invalid(self):

        registry = make_registry(
            lambda state: ValidationResult(False, "too few", {})
        )

        _, record = run(registry)

        step = record["steps"][1]

        self.assertEqual(step["status"], "invalid")
        self.assertEqual(step["error"], "too few")
        self.assertEqual(record["failed_step"], "b")
        self.assertEqual(record["error"], "too few")
        self.assertFalse(record["success"])

    def test_invalid_node_blocks_its_dependents(self):

        registry = AgentRegistry()

        registry.register(AgentSpec(
            name="a",
            description="d",
            produces=("x",),
            factory=lambda **p: FakeAgent(name="a"),
            validator=lambda s: ValidationResult(False, "bad", {})
        ))
        registry.register(AgentSpec(
            name="b",
            description="d",
            requires=("x",),
            produces=("goal",),
            factory=lambda **p: FakeAgent(name="b")
        ))

        _, record = run(registry)

        self.assertEqual(record["skipped"], ["b"])
        self.assertEqual(len(record["steps"]), 1)

    def test_validate_false_ignores_validators(self):

        registry = make_registry(
            lambda state: ValidationResult(False, "too few", {})
        )

        _, record = run(registry, validate=False)

        self.assertEqual(record["steps"][1]["status"], "ok")
        self.assertIsNone(record["steps"][1]["validation"])

    def test_a_crashing_validator_rejects_the_node(self):

        def broken(state):
            raise RuntimeError("oops")

        _, record = run(make_registry(broken))

        step = record["steps"][1]

        self.assertEqual(step["status"], "invalid")
        self.assertIn("Validator error", step["error"])
        self.assertIn("oops", step["error"])


class RealAgentsValidationTests(unittest.TestCase):
    """Real agents with fake clients: the fake gives only 2 leads."""

    def wired(self, min_leads):

        original = build_default_registry(min_leads=min_leads)

        registry = AgentRegistry()

        gemini = RoutingGemini()

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
                validator=spec.validator
            ))

        return registry

    def run_scores(self, min_leads):

        registry = self.wired(min_leads)

        plan = Planner(registry).plan("scores")

        return asyncio.run(
            Executor(registry).run(plan, "A billing tool.")
        )

    def test_too_few_leads_stops_the_pipeline_before_enrichment(self):

        _, record = self.run_scores(min_leads=3)

        self.assertFalse(record["success"])
        self.assertEqual(record["failed_step"], "lead_discovery_agent")
        self.assertEqual(
            record["steps"][-1]["status"],
            "invalid"
        )
        self.assertEqual(
            record["skipped"],
            ["enrichment_agent", "scoring_agent"]
        )

        # Enrichment and scoring never ran, so they cost nothing.
        self.assertEqual(record["agent_calls"], 2)
        self.assertEqual(record["llm_calls"], 2)
        self.assertEqual(record["search_calls"], 3)

    def test_enough_leads_lets_everything_run(self):

        _, record = self.run_scores(min_leads=2)

        self.assertTrue(record["success"])
        self.assertEqual(record["skipped"], [])
        self.assertEqual(record["agent_calls"], 4)

        for step in record["steps"]:
            self.assertTrue(step["validation"]["passed"], step)


if __name__ == "__main__":
    unittest.main()
