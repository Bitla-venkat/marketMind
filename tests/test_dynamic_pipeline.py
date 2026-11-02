"""
Offline tests for pipelines/dynamic_pipeline.py, the models_used field
and the extended experiments/run_baseline.py summary.

Run from the project root:

    python -m unittest tests.test_dynamic_pipeline -v
"""

import asyncio
import os
import unittest
from dataclasses import replace

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from experiments.run_baseline import (
    ALL_VARIANTS,
    build_pipeline,
    summarise_runs
)
from pipelines.dynamic_pipeline import (
    DYNAMIC_VARIANTS,
    DynamicPipeline,
    build_dynamic_pipeline
)
from pipelines.fixed_pipeline import build_fixed_pipeline, models_used

from tests.test_executor_recovery import FlakyLeadsGemini
from tests.test_pipeline_integration import FakeSearchTool, wire_fakes


def wire_dynamic(pipeline, gemini):
    """Give every agent the pipeline builds fake clients."""

    registry = pipeline.registry

    for name, spec in list(registry._specs.items()):

        def factory(_spec=spec, **params):

            agent = _spec.factory(**params)

            agent.client = gemini

            if hasattr(agent, "search_tool"):
                agent.search_tool = FakeSearchTool()

            return agent

        registry._specs[name] = replace(spec, factory=factory)


def run_dynamic(variant, gemini):

    pipeline = build_dynamic_pipeline(variant)

    wire_dynamic(pipeline, gemini)

    return asyncio.run(pipeline.run("A billing tool.", label="x"))


class BuildTests(unittest.TestCase):

    def test_variants_configure_validation_and_recovery(self):

        full = build_dynamic_pipeline("dynamic")
        no_recovery = build_dynamic_pipeline("dynamic-no-recovery")
        plain = build_dynamic_pipeline("dynamic-plain")

        self.assertTrue(full.executor.validate)
        self.assertIsNotNone(full.executor.recovery)

        self.assertTrue(no_recovery.executor.validate)
        self.assertIsNone(no_recovery.executor.recovery)

        self.assertFalse(plain.executor.validate)
        self.assertIsNone(plain.executor.recovery)

        for variant in DYNAMIC_VARIANTS:
            self.assertEqual(
                build_dynamic_pipeline(variant).name,
                variant
            )

    def test_plan_is_the_four_agent_chain(self):

        pipeline = build_dynamic_pipeline()

        self.assertEqual(
            [node.id for node in pipeline.plan.nodes],
            [
                "icp_agent",
                "lead_discovery_agent",
                "enrichment_agent",
                "scoring_agent"
            ]
        )

    def test_options_are_passed_through(self):

        pipeline = build_dynamic_pipeline(
            "dynamic",
            min_leads=5,
            max_recoveries_per_run=1
        )

        self.assertEqual(
            pipeline.executor.recovery.max_recoveries_per_run,
            1
        )

        validator = pipeline.registry.get(
            "lead_discovery_agent"
        ).validator

        from core.state import AgentState

        state = AgentState("s")
        state.leads = ["l"] * 4

        result = validator(state)

        self.assertFalse(result.passed)
        self.assertEqual(result.metrics["min_leads"], 5)

    def test_unknown_variant_rejected(self):

        with self.assertRaises(ValueError):
            build_dynamic_pipeline("nope")

    def test_build_pipeline_maps_every_variant_name(self):

        names = {
            variant: build_pipeline(variant).name
            for variant in ALL_VARIANTS
        }

        self.assertEqual(names["narrow"], "fixed-narrow")
        self.assertEqual(names["wide"], "fixed-wide")
        self.assertEqual(names["dynamic"], "dynamic")
        self.assertEqual(len(names), 5)


class EndToEndTests(unittest.TestCase):

    def test_healthy_run_matches_the_fixed_pipeline_cost(self):

        _, record = run_dynamic(
            "dynamic", FlakyLeadsGemini(first=3, later=3)
        )

        self.assertTrue(record["success"])
        self.assertEqual(record["pipeline"], "dynamic")
        self.assertEqual(record["recoveries"], [])
        self.assertEqual(record["agent_calls"], 4)

    def test_a_thin_result_is_recovered(self):

        gemini = FlakyLeadsGemini(first=1, later=3)

        _, record = run_dynamic("dynamic", gemini)

        self.assertTrue(record["success"])
        self.assertEqual(gemini.discovery_calls, 2)
        self.assertEqual(len(record["recoveries"]), 1)

        # The retry cost one more discovery call.
        self.assertEqual(record["agent_calls"], 5)

    def test_without_recovery_a_thin_result_stops_the_run(self):

        _, record = run_dynamic(
            "dynamic-no-recovery",
            FlakyLeadsGemini(first=1, later=3)
        )

        self.assertFalse(record["success"])
        self.assertEqual(record["failed_step"], "lead_discovery_agent")

    def test_plain_variant_accepts_the_thin_result(self):

        _, record = run_dynamic(
            "dynamic-plain",
            FlakyLeadsGemini(first=1, later=3)
        )

        # No validator: runs on with one lead, like the fixed pipeline.
        self.assertTrue(record["success"])
        self.assertEqual(record["leads_found"], 1)

    def test_fixed_and_dynamic_records_share_their_fields(self):

        fixed_pipeline = build_fixed_pipeline("narrow")

        wire_fakes(fixed_pipeline, FlakyLeadsGemini(3, 3))

        _, fixed = asyncio.run(fixed_pipeline.run("A billing tool."))

        _, dynamic = run_dynamic("dynamic", FlakyLeadsGemini(3, 3))

        self.assertEqual(set(fixed) - set(dynamic), set())
        self.assertEqual(
            set(dynamic) - set(fixed),
            {"skipped", "recoveries", "plan", "max_concurrency"}
        )


class ModelsUsedTests(unittest.TestCase):

    def test_counts_calls_per_model(self):

        calls = [
            {"model": "a"}, {"model": "b"}, {"model": "a"}
        ]

        self.assertEqual(models_used(calls), {"a": 2, "b": 1})
        self.assertEqual(models_used([]), {})

    def test_both_pipelines_record_which_model_answered(self):

        fixed_pipeline = build_fixed_pipeline("narrow")

        wire_fakes(fixed_pipeline, FlakyLeadsGemini(3, 3))

        _, fixed = asyncio.run(fixed_pipeline.run("x"))

        _, dynamic = run_dynamic("dynamic", FlakyLeadsGemini(3, 3))

        for record in (fixed, dynamic):

            self.assertGreater(len(record["models_used"]), 0)
            self.assertEqual(
                sum(record["models_used"].values()),
                record["llm_calls"]
            )


def make_record(pipeline="dynamic", scored=3, success=True,
                recoveries=0, models=None):

    return {
        "pipeline": pipeline,
        "success": success,
        "failed_step": None if success else "x",
        "leads_scored": scored,
        "latency_s": 10.0,
        "llm_calls": 5,
        "llm_failed_attempts": 0,
        "input_tokens": 100,
        "output_tokens": 10,
        "search_calls": 4,
        "recoveries": [{}] * recoveries,
        "models_used": models or {}
    }


class SummaryTests(unittest.TestCase):

    def test_stricter_success_rate_and_recoveries(self):

        records = [
            make_record(scored=3, recoveries=0),
            make_record(scored=1, recoveries=2),
            make_record(scored=0, success=False, recoveries=1),
            make_record(scored=5, recoveries=1)
        ]

        summary = summarise_runs(records)["dynamic"]

        self.assertEqual(summary["success_rate"], 0.75)

        # Only the runs that scored 3 or more count.
        self.assertEqual(summary["success_rate_min3_leads"], 0.5)
        self.assertEqual(summary["mean_recoveries"], 1.0)

    def test_models_are_summed_and_old_records_still_work(self):

        records = [
            make_record(models={"m1": 3, "m2": 1}),
            make_record(models={"m1": 2})
        ]

        old = make_record()

        del old["recoveries"]
        del old["models_used"]

        records.append(old)

        summary = summarise_runs(records)["dynamic"]

        self.assertEqual(summary["models_used"], {"m1": 5, "m2": 1})
        self.assertEqual(summary["runs"], 3)


if __name__ == "__main__":
    unittest.main()
