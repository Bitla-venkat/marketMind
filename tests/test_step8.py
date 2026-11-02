"""
Offline tests for: frozen ICPs, running pipelines from a given ICP,
failure kinds, target-vs-minimum validation and the new summary fields.

Run from the project root:

    python -m unittest tests.test_step8 -v
"""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.executor import Executor
from core.models import ICP, Lead
from core.planner import Planner
from core.registry import AgentRegistry, AgentSpec
from core.state import AgentState
from core.validation import (
    ValidationResult,
    make_leads_validator
)
from experiments.frozen_icps import get_frozen_icps
from experiments.run_baseline import summarise_runs
from pipelines.fixed_pipeline import (
    FixedPipeline,
    build_fixed_pipeline,
    count_failure_kinds,
    merge_counts
)

from tests.test_dynamic_pipeline import wire_dynamic
from tests.test_executor_recovery import FlakyLeadsGemini
from tests.test_fixed_pipeline import FakeAgent as FixedFakeAgent
from tests.test_pipeline_integration import wire_fakes
from pipelines.dynamic_pipeline import build_dynamic_pipeline


def make_icp(industry="Retail"):

    return ICP(
        target_industries=[industry],
        target_locations=["India"],
        company_size="SMB"
    )


STARTUPS = [
    {"id": "s1", "description": "first startup"},
    {"id": "s2", "description": "second startup"}
]


class FrozenIcpTests(unittest.TestCase):

    def setUp(self):

        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "frozen.json"
        self.calls = []

        async def generate(description):

            self.calls.append(description)

            return make_icp(f"industry-{len(self.calls)}")

        self.generate = generate

    def tearDown(self):

        self.dir.cleanup()

    def get(self, startups=STARTUPS):

        return asyncio.run(
            get_frozen_icps(
                startups,
                path=self.path,
                generate=self.generate
            )
        )

    def test_first_call_generates_and_saves(self):

        icps = self.get()

        self.assertEqual(set(icps), {"s1", "s2"})
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(self.path.exists())

    def test_second_call_reuses_the_saved_icps(self):

        first = self.get()
        second = self.get()

        self.assertEqual(len(self.calls), 2)
        self.assertEqual(first["s1"], second["s1"])
        self.assertEqual(first["s2"], second["s2"])

    def test_only_missing_startups_are_generated(self):

        self.get(STARTUPS[:1])

        self.get()

        self.assertEqual(
            self.calls,
            ["first startup", "second startup"]
        )

    def test_changed_description_regenerates_that_entry(self):

        first = self.get()

        changed = [
            {"id": "s1", "description": "a different description"},
            STARTUPS[1]
        ]

        second = self.get(changed)

        self.assertEqual(len(self.calls), 3)
        self.assertNotEqual(first["s1"], second["s1"])
        self.assertEqual(first["s2"], second["s2"])

    def test_corrupt_file_is_regenerated(self):

        self.path.write_text("{not json")

        icps = self.get()

        self.assertEqual(len(icps), 2)
        self.assertEqual(len(self.calls), 2)

        json.loads(self.path.read_text())

    def test_invalid_entry_is_regenerated(self):

        self.get(STARTUPS[:1])

        data = json.loads(self.path.read_text())
        data["s1"]["icp"] = {"target_industries": "oops"}
        self.path.write_text(json.dumps(data))

        self.get(STARTUPS[:1])

        self.assertEqual(len(self.calls), 2)


class PipelinesFromAGivenIcpTests(unittest.TestCase):

    def test_fixed_pipeline_skips_the_icp_step(self):

        icp = make_icp("Frozen")

        pipeline = build_fixed_pipeline("narrow")

        gemini = FlakyLeadsGemini(3, 3)

        wire_fakes(pipeline, gemini)

        state, record = asyncio.run(
            pipeline.run("A billing tool.", label="x", icp=icp)
        )

        self.assertIs(state.icp, icp)
        self.assertTrue(record["frozen_icp"])
        self.assertNotIn("ICP", gemini.schemas)
        self.assertEqual(
            [s["agent"] for s in record["steps"]],
            [
                "lead_discovery_agent",
                "enrichment_agent",
                "scoring_agent"
            ]
        )
        self.assertTrue(record["success"])

    def test_fixed_pipeline_without_icp_still_runs_it(self):

        pipeline = build_fixed_pipeline("narrow")

        wire_fakes(pipeline, FlakyLeadsGemini(3, 3))

        _, record = asyncio.run(pipeline.run("A billing tool."))

        self.assertFalse(record["frozen_icp"])
        self.assertEqual(record["steps"][0]["agent"], "icp_agent")

    def test_dynamic_pipeline_plans_without_the_icp_agent(self):

        icp = make_icp("Frozen")

        pipeline = build_dynamic_pipeline("dynamic")

        wire_dynamic(pipeline, FlakyLeadsGemini(3, 3))

        state, record = asyncio.run(
            pipeline.run("A billing tool.", label="x", icp=icp)
        )

        self.assertIs(state.icp, icp)
        self.assertTrue(record["frozen_icp"])
        self.assertTrue(record["success"])

        self.assertEqual(
            [node["id"] for node in record["plan"]["nodes"]],
            [
                "lead_discovery_agent",
                "enrichment_agent",
                "scoring_agent"
            ]
        )

    def test_both_pipelines_cost_the_same_when_healthy(self):

        icp = make_icp()

        fixed = build_fixed_pipeline("narrow")
        wire_fakes(fixed, FlakyLeadsGemini(3, 3))

        _, fixed_record = asyncio.run(
            fixed.run("A billing tool.", icp=icp)
        )

        dynamic = build_dynamic_pipeline("dynamic")
        wire_dynamic(dynamic, FlakyLeadsGemini(3, 3))

        _, dynamic_record = asyncio.run(
            dynamic.run("A billing tool.", icp=icp)
        )

        for key in ("agent_calls", "llm_calls", "search_calls",
                    "leads_found", "leads_scored"):
            self.assertEqual(
                fixed_record[key],
                dynamic_record[key],
                key
            )

    def test_executor_places_initial_artifacts_in_the_state(self):

        registry = AgentRegistry()

        class Reader:

            name = "reader"

            async def run(self, task, state):
                state.set_artifact("out", [state.get_artifact("seed")])

        registry.register(AgentSpec(
            name="reader",
            description="d",
            requires=("seed",),
            produces=("out",),
            factory=lambda **p: Reader()
        ))

        plan = Planner(registry).plan("out", available={"seed"})

        state, record = asyncio.run(
            Executor(registry).run(plan, "s", initial={"seed": 42})
        )

        self.assertEqual(state.get_artifact("out"), [42])
        self.assertEqual(record["failed_step"], None)


class FailureKindTests(unittest.TestCase):

    def test_kind_is_the_code_else_the_error_type(self):

        failures = [
            {"code": 429, "type": "ClientError"},
            {"code": 429, "type": "ClientError"},
            {"code": None, "type": "TimeoutError"},
            {"type": "ReadError"},
            {}
        ]

        self.assertEqual(
            count_failure_kinds(failures),
            {"429": 2, "TimeoutError": 1, "ReadError": 1, "unknown": 1}
        )

    def test_merge_counts(self):

        self.assertEqual(
            merge_counts([{"a": 1}, {"a": 2, "b": 1}, {}]),
            {"a": 3, "b": 1}
        )
        self.assertEqual(merge_counts([]), {})

    def test_fixed_record_counts_recovered_and_fatal_failures(self):

        class Fatal(FixedFakeAgent):

            async def run(self, task, state):

                error = RuntimeError("down")
                error.llm_failures = [
                    {"code": None, "type": "TimeoutError"}
                ]

                raise error

        steps = [
            (FixedFakeAgent(
                "a", llm_calls=2, recovered_failures=1
            ), None),
            (Fatal("b"), None)
        ]

        _, record = asyncio.run(FixedPipeline("p", steps).run("x"))

        self.assertEqual(
            record["failure_kinds"],
            {"503": 2, "TimeoutError": 1}
        )
        self.assertEqual(record["llm_failed_attempts"], 3)

    def test_executor_record_counts_failure_kinds(self):

        class Agent:

            name = "a"

            async def run(self, task, state):

                state.llm_calls.append({
                    "agent": "a",
                    "model": "m",
                    "attempts": 3,
                    "latency_s": 1.0,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "thinking_tokens": 0,
                    "failures": [
                        {"code": 429}, {"type": "TimeoutError"}
                    ]
                })

                state.set_artifact("out", ["x"])

        registry = AgentRegistry()
        registry.register(AgentSpec(
            name="a",
            description="d",
            produces=("out",),
            factory=lambda **p: Agent()
        ))

        _, record = asyncio.run(
            Executor(registry).run(
                Planner(registry).plan("out"), "s"
            )
        )

        self.assertEqual(
            record["failure_kinds"],
            {"429": 1, "TimeoutError": 1}
        )
        self.assertEqual(record["llm_failed_attempts"], 2)


class TargetVersusMinimumTests(unittest.TestCase):

    def state_with(self, count):

        state = AgentState("s")
        state.leads = [
            Lead(company_name=str(i), reason_for_match="r")
            for i in range(count)
        ]

        return state

    def test_acceptable_between_floor_and_target(self):

        validator = make_leads_validator(min_leads=3, floor=1)

        zero = validator(self.state_with(0))
        two = validator(self.state_with(2))
        three = validator(self.state_with(3))

        self.assertFalse(zero.passed)
        self.assertFalse(zero.can_continue)

        self.assertFalse(two.passed)
        self.assertTrue(two.can_continue)

        self.assertTrue(three.passed)
        self.assertTrue(three.can_continue)

    def test_floor_equal_to_target_means_no_tolerance(self):

        validator = make_leads_validator(min_leads=3, floor=3)

        self.assertFalse(validator(self.state_with(2)).can_continue)

    def test_floor_above_target_is_rejected(self):

        with self.assertRaises(ValueError):
            make_leads_validator(min_leads=2, floor=3)

    def test_result_without_acceptable_follows_passed(self):

        self.assertTrue(ValidationResult(True).can_continue)
        self.assertFalse(ValidationResult(False).can_continue)
        self.assertTrue(
            ValidationResult(False, acceptable=True).can_continue
        )

    def test_to_dict_carries_acceptable(self):

        data = make_leads_validator(3, 1)(self.state_with(2)).to_dict()

        self.assertFalse(data["passed"])
        self.assertTrue(data["acceptable"])


def make_record(pipeline="p", latency=10.0, kinds=None, degraded=None,
                success=True, scored=3):

    return {
        "pipeline": pipeline,
        "success": success,
        "failed_step": None if success else "x",
        "leads_scored": scored,
        "latency_s": latency,
        "agent_calls": 4,
        "llm_calls": 5,
        "llm_failed_attempts": 0,
        "input_tokens": 100,
        "output_tokens": 10,
        "search_calls": 4,
        "failure_kinds": kinds or {},
        "degraded": degraded or []
    }


class SummaryTests(unittest.TestCase):

    def test_median_max_and_failure_kinds(self):

        records = [
            make_record(latency=10, kinds={"429": 4}),
            make_record(latency=20, kinds={"429": 1, "TimeoutError": 2}),
            make_record(latency=1100, degraded=["lead_discovery_agent"])
        ]

        summary = summarise_runs(records)["p"]

        # One slow run must not hide the typical one.
        self.assertEqual(summary["median_latency_s"], 20.0)
        self.assertEqual(summary["max_latency_s"], 1100)
        self.assertGreater(summary["mean_latency_s"], 300)
        self.assertEqual(
            summary["failure_kinds"],
            {"429": 5, "TimeoutError": 2}
        )
        self.assertEqual(summary["runs_with_degraded_step"], 1)
        self.assertEqual(summary["mean_agent_calls"], 4.0)


if __name__ == "__main__":
    unittest.main()
