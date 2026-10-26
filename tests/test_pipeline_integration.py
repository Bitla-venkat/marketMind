"""
Offline end-to-end test: the REAL agents wired into the fixed pipeline,
with Gemini and Tavily replaced by fakes.

This checks that data flows correctly from one agent to the next
(ICP -> leads -> enrichment -> scores) and that the run record counts
match. Run from the project root:

    python -m unittest tests.test_pipeline_integration -v
"""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.models import (
    ICP,
    EnrichmentExtract,
    Lead,
    LeadList,
    LeadScore,
    LeadScoreList
)
from pipelines.fixed_pipeline import build_fixed_pipeline


RESULTS = [
    {"url": "https://a.example", "title": "A", "content": "a"},
    {"url": "https://b.example", "title": "B", "content": "b"}
]


class FakeResponse:

    def __init__(self, parsed):
        self.parsed = parsed
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=10,
            thoughts_token_count=0
        )


class RoutingGemini:
    """Answers each call according to the response schema asked for."""

    def __init__(self):
        self.schemas = []

    async def _generate(self, model, contents, config):

        schema = config["response_schema"]

        self.schemas.append(schema.__name__)

        if schema is ICP:

            return FakeResponse(ICP(
                target_industries=["Retail", "Trading", "Wholesale"],
                target_locations=["India"],
                company_size="SMB"
            ))

        if schema is LeadList:

            return FakeResponse(LeadList(leads=[
                Lead(
                    company_name="Alpha",
                    reason_for_match="fits",
                    source_url="https://a.example"
                ),
                Lead(
                    company_name="Beta",
                    reason_for_match="fits",
                    source_url="https://b.example"
                )
            ]))

        if schema is EnrichmentExtract:

            return FakeResponse(EnrichmentExtract(
                website="https://a.example"
            ))

        if schema is LeadScoreList:

            return FakeResponse(LeadScoreList(scores=[
                LeadScore(
                    company_name="Alpha",
                    score=80,
                    rationale="strong"
                ),
                LeadScore(
                    company_name="Beta",
                    score=55,
                    rationale="ok"
                )
            ]))

        raise AssertionError(f"Unexpected schema {schema}")

    @property
    def aio(self):

        return SimpleNamespace(
            models=SimpleNamespace(generate_content=self._generate)
        )


class FakeSearchTool:

    def __init__(self):
        self.search_count = 0

    async def async_search(self, query, max_results=5):

        self.search_count += 1

        return list(RESULTS)


def wire_fakes(pipeline, gemini):
    """Replace every agent's real clients with fakes."""

    for agent, _ in pipeline.steps:

        agent.client = gemini

        if hasattr(agent, "search_tool"):
            agent.search_tool = FakeSearchTool()


class PipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):

    async def test_data_flows_through_all_four_agents(self):

        pipeline = build_fixed_pipeline("narrow")

        gemini = RoutingGemini()

        wire_fakes(pipeline, gemini)

        state, record = await pipeline.run(
            "A billing tool for retailers.",
            label="gst"
        )

        # Data produced by each agent reached the next.
        self.assertEqual(
            state.icp.target_industries,
            ["Retail", "Trading", "Wholesale"]
        )
        self.assertEqual(
            [lead.company_name for lead in state.leads],
            ["Alpha", "Beta"]
        )
        self.assertEqual(
            [e.status for e in state.enriched_leads],
            ["ok", "ok"]
        )
        self.assertEqual(list(state.scores), ["Alpha", "Beta"])
        self.assertEqual(state.scores["Alpha"]["score"], 80)

        # Gemini was asked for the four things, in order.
        self.assertEqual(
            gemini.schemas,
            [
                "ICP",
                "LeadList",
                "EnrichmentExtract",
                "EnrichmentExtract",
                "LeadScoreList"
            ]
        )

        # Run record.
        self.assertTrue(record["success"])
        self.assertIsNone(record["failed_step"])
        self.assertEqual(record["pipeline"], "fixed-narrow")
        self.assertEqual(record["label"], "gst")
        self.assertEqual(record["agent_calls"], 4)
        self.assertEqual(record["llm_calls"], 5)

        # Narrow: 3 discovery searches + 1 per lead for enrichment.
        self.assertEqual(record["search_calls"], 5)

        self.assertEqual(record["leads_found"], 2)
        self.assertEqual(record["leads_enriched_ok"], 2)
        self.assertEqual(record["leads_scored"], 2)
        self.assertEqual(record["input_tokens"], 500)

    async def test_wide_variant_searches_more(self):

        pipeline = build_fixed_pipeline("wide")

        wire_fakes(pipeline, RoutingGemini())

        _, record = await pipeline.run("A billing tool.", label="x")

        # Wide: 3 industries x 3 queries + 1 per lead for enrichment.
        self.assertEqual(record["search_calls"], 9 + 2)
        self.assertTrue(record["success"])

    async def test_zero_leads_ends_the_fixed_pipeline_in_failure(self):

        class NoLeadsGemini(RoutingGemini):

            async def _generate(self, model, contents, config):

                if config["response_schema"] is LeadList:

                    self.schemas.append("LeadList")

                    return FakeResponse(LeadList(leads=[]))

                return await super()._generate(
                    model, contents, config
                )

        pipeline = build_fixed_pipeline("narrow")

        wire_fakes(pipeline, NoLeadsGemini())

        state, record = await pipeline.run("A billing tool.")

        # Discovery finished with nothing; the fixed pipeline does not
        # notice, and enrichment then fails for lack of input.
        self.assertFalse(record["success"])
        self.assertEqual(record["failed_step"], "enrichment_agent")
        self.assertEqual(record["leads_found"], 0)
        self.assertEqual(record["leads_scored"], 0)
        self.assertEqual(record["steps"][1]["status"], "ok")
        self.assertEqual(record["agent_calls"], 3)


if __name__ == "__main__":
    unittest.main()
