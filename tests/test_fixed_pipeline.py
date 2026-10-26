"""
Offline tests for pipelines/fixed_pipeline.py.

Uses fake agents, so no keys or network are needed.
Run from the project root:

    python -m unittest tests.test_fixed_pipeline -v
"""

import os
import unittest
from types import SimpleNamespace

# Building the real pipeline only checks that keys exist.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.models import EnrichedLead, Lead
from pipelines.fixed_pipeline import (
    FixedPipeline,
    VARIANTS,
    build_fixed_pipeline,
    is_success
)


class FakeAgent:
    """
    Stand-in for an agent.

    `effect` is called as effect(state) when the agent runs and may
    raise. `llm_calls` is how many LLM call records to append, and
    `searches` how many searches the agent's search tool reports.
    """

    def __init__(
        self,
        name,
        effect=None,
        llm_calls=0,
        searches=0,
        recovered_failures=0
    ):
        self.name = name
        self.effect = effect
        self.llm_calls = llm_calls
        self.searches = searches
        self.recovered_failures = recovered_failures
        self.received_task = "unset"

        self.search_tool = SimpleNamespace(search_count=0)

    async def run(self, task, state):

        self.received_task = task

        if hasattr(self, "search_tool"):
            self.search_tool.search_count += self.searches

        for _ in range(self.llm_calls):

            state.llm_calls.append({
                "agent": self.name,
                "model": "m",
                "attempts": 1 + self.recovered_failures,
                "latency_s": 0.1,
                "input_tokens": 100,
                "output_tokens": 10,
                "thinking_tokens": 0,
                "failures": [
                    {"code": 503}
                ] * self.recovered_failures
            })

        if self.effect:
            self.effect(state)


def make_lead(name="Acme"):

    return Lead(company_name=name, reason_for_match="fits")


def add_leads(state):

    state.leads = [make_lead("A"), make_lead("B")]


def add_enriched(state):

    state.enriched_leads = [
        EnrichedLead(lead=make_lead("A"), status="ok"),
        EnrichedLead(lead=make_lead("B"), status="llm_error")
    ]


def add_scores(state):

    state.scores = {"A": {"score": 80}}


def boom(_state):

    raise RuntimeError("tavily down")


def full_steps():

    return [
        (FakeAgent("icp", llm_calls=1), {"type": "icp"}),
        (
            FakeAgent(
                "discovery",
                effect=add_leads,
                llm_calls=1,
                searches=3
            ),
            {"type": "discover"}
        ),
        (
            FakeAgent(
                "enrichment",
                effect=add_enriched,
                llm_calls=2,
                searches=2,
                recovered_failures=1
            ),
            None
        ),
        (
            FakeAgent("scoring", effect=add_scores, llm_calls=1),
            {"type": "score"}
        )
    ]


class FixedPipelineTests(unittest.IsolatedAsyncioTestCase):

    async def test_runs_every_step_in_order_with_its_task(self):

        steps = full_steps()

        pipeline = FixedPipeline("fixed-test", steps)

        state, record = await pipeline.run("a startup", label="s1")

        self.assertEqual(
            [step["agent"] for step in record["steps"]],
            ["icp", "discovery", "enrichment", "scoring"]
        )
        self.assertEqual(steps[0][0].received_task, {"type": "icp"})
        self.assertIsNone(steps[2][0].received_task)
        self.assertEqual(record["pipeline"], "fixed-test")
        self.assertEqual(record["label"], "s1")
        self.assertEqual(state.startup_description, "a startup")

    async def test_successful_run_totals(self):

        _, record = await FixedPipeline(
            "p", full_steps()
        ).run("x")

        self.assertTrue(record["success"])
        self.assertIsNone(record["failed_step"])
        self.assertEqual(record["agent_calls"], 4)
        self.assertEqual(record["llm_calls"], 5)
        self.assertEqual(record["input_tokens"], 500)
        self.assertEqual(record["output_tokens"], 50)
        self.assertEqual(record["search_calls"], 5)
        self.assertEqual(record["leads_found"], 2)
        self.assertEqual(record["leads_enriched_ok"], 1)
        self.assertEqual(record["leads_scored"], 1)

        # Two enrichment calls each recovered from one failed attempt.
        self.assertEqual(record["llm_failed_attempts"], 2)

        self.assertGreaterEqual(record["latency_s"], 0)

    async def test_per_step_costs_are_separated(self):

        _, record = await FixedPipeline(
            "p", full_steps()
        ).run("x")

        by_agent = {s["agent"]: s for s in record["steps"]}

        self.assertEqual(by_agent["icp"]["llm_calls"], 1)
        self.assertEqual(by_agent["icp"]["search_calls"], 0)
        self.assertEqual(by_agent["discovery"]["search_calls"], 3)
        self.assertEqual(by_agent["enrichment"]["llm_calls"], 2)
        self.assertEqual(by_agent["enrichment"]["search_calls"], 2)
        self.assertEqual(by_agent["scoring"]["llm_calls"], 1)

    async def test_failure_stops_the_pipeline(self):

        steps = [
            (FakeAgent("icp", llm_calls=1), None),
            (FakeAgent("discovery", effect=boom, searches=1), None),
            (FakeAgent("enrichment", llm_calls=1), None)
        ]

        _, record = await FixedPipeline("p", steps).run("x")

        self.assertFalse(record["success"])
        self.assertEqual(record["failed_step"], "discovery")
        self.assertIn("tavily down", record["error"])
        self.assertEqual(
            [step["agent"] for step in record["steps"]],
            ["icp", "discovery"]
        )

        # The failed step still counts: it was called and cost a search.
        self.assertEqual(record["agent_calls"], 2)
        self.assertEqual(record["search_calls"], 1)
        self.assertEqual(record["steps"][1]["status"], "failed")

        # The agent after the failure never ran.
        self.assertEqual(steps[2][0].received_task, "unset")

    async def test_failed_llm_attempts_on_exception_are_counted(self):

        def llm_failure(_state):

            error = RuntimeError("model down")
            error.llm_failures = [{"code": 503}, {"code": 503}]

            raise error

        steps = [(FakeAgent("icp", effect=llm_failure), None)]

        _, record = await FixedPipeline("p", steps).run("x")

        self.assertEqual(record["llm_failed_attempts"], 2)

    async def test_zero_scores_is_not_a_success(self):

        steps = [
            (FakeAgent("icp"), None),
            (FakeAgent("scoring"), None)
        ]

        _, record = await FixedPipeline("p", steps).run("x")

        # Nothing raised, but nothing was scored.
        self.assertIsNone(record["failed_step"])
        self.assertEqual(record["leads_scored"], 0)
        self.assertFalse(record["success"])

    async def test_agents_without_a_search_tool_are_fine(self):

        agent = FakeAgent("icp", llm_calls=1)
        del agent.search_tool

        _, record = await FixedPipeline(
            "p", [(agent, None)]
        ).run("x")

        self.assertIsNone(record["failed_step"])
        self.assertEqual(record["steps"][0]["status"], "ok")
        self.assertEqual(record["search_calls"], 0)


class IsSuccessTests(unittest.TestCase):

    def test_definition(self):

        self.assertTrue(
            is_success({"failed_step": None, "leads_scored": 1})
        )
        self.assertFalse(
            is_success({"failed_step": None, "leads_scored": 0})
        )
        self.assertFalse(
            is_success({"failed_step": "icp", "leads_scored": 3})
        )


class BuildFixedPipelineTests(unittest.TestCase):

    def test_builds_the_four_real_agents_in_order(self):

        pipeline = build_fixed_pipeline("narrow")

        self.assertEqual(pipeline.name, "fixed-narrow")
        self.assertEqual(
            [agent.name for agent, _ in pipeline.steps],
            [
                "icp_agent",
                "lead_discovery_agent",
                "enrichment_agent",
                "scoring_agent"
            ]
        )

    def test_variants_set_discovery_settings(self):

        for variant, settings in VARIANTS.items():

            discovery = build_fixed_pipeline(variant).steps[1][0]

            self.assertEqual(
                discovery.max_industries,
                settings["max_industries"]
            )
            self.assertEqual(
                discovery.results_per_query,
                settings["results_per_query"]
            )

    def test_unknown_variant_is_rejected(self):

        with self.assertRaises(ValueError):
            build_fixed_pipeline("nope")


if __name__ == "__main__":
    unittest.main()
