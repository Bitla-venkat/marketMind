"""
Offline tests for agents/enrichment_agent.py.

No real API calls: Gemini and Tavily are replaced with fakes.
Run from the project root:

    python -m unittest tests.test_enrichment -v
"""

import os
import unittest
from types import SimpleNamespace

# The agent constructor only checks that keys exist.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from agents.enrichment_agent import EnrichmentAgent
from core.models import (
    EnrichedLead,
    EnrichmentExtract,
    EvidenceItem,
    Lead
)
from core.state import AgentState
from experiments.probe_enrichment import fill_rates, summarise


class FakeAPIError(Exception):

    def __init__(self, code, message=""):
        super().__init__(f"{code} {message}")
        self.code = code


class FakeResponse:

    def __init__(self, parsed):
        self.parsed = parsed
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=200,
            candidates_token_count=40,
            thoughts_token_count=0
        )


class FakeGemini:
    """
    Replays one scripted outcome per Gemini call.

    Each item is an Exception (raised), or a value used as the
    parsed result (None means "could not be parsed").
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

        self.aio = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=self._generate
            )
        )

    async def _generate(self, model, contents, config):

        self.calls += 1

        item = self.script.pop(0)

        if isinstance(item, Exception):
            raise item

        return FakeResponse(item)


class FakeSearch:
    """
    Replays one scripted outcome per search: an Exception
    (raised) or a list of results.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def async_search(self, query, max_results=5):

        self.calls.append((query, max_results))

        item = self.script.pop(0)

        if isinstance(item, Exception):
            raise item

        return item


RESULTS = [
    {"url": "https://acme.example/about", "title": "About", "content": "x"},
    {"url": "https://linkedin.example/acme", "title": "LI", "content": "y"}
]


def make_lead(name="Acme"):

    return Lead(
        company_name=name,
        reason_for_match="matches ICP",
        source_url="https://acme.example"
    )


def make_extract(**kwargs):

    return EnrichmentExtract(**kwargs)


def make_agent(search, gemini, **kwargs):

    agent = EnrichmentAgent(**kwargs)
    agent.search_tool = search
    agent.client = gemini

    return agent


def make_state(leads):

    state = AgentState("test startup")
    state.leads = leads

    return state


class EnrichmentAgentTests(unittest.IsolatedAsyncioTestCase):

    async def test_ok_returns_typed_result(self):

        extract = make_extract(
            website="https://acme.example",
            founders=["A. Founder"],
            evidence=[
                EvidenceItem(
                    claim="Founded by A. Founder",
                    url="https://acme.example/about/"
                ),
                EvidenceItem(
                    claim="Invented claim",
                    url="https://invented.example"
                )
            ]
        )

        state = make_state([make_lead()])

        agent = make_agent(
            FakeSearch([RESULTS]),
            FakeGemini([extract])
        )

        enriched = await agent.run(None, state)

        self.assertEqual(len(enriched), 1)
        self.assertIsInstance(enriched[0], EnrichedLead)
        self.assertEqual(enriched[0].status, "ok")
        self.assertEqual(
            enriched[0].details.website,
            "https://acme.example"
        )
        self.assertEqual(enriched[0].lead.company_name, "Acme")
        self.assertEqual(state.enriched_leads, enriched)

        diag = state.diagnostics["enrichment_agent"]

        self.assertEqual(diag["status_counts"], {"ok": 1})
        self.assertEqual(diag["evidence_total"], 2)

        # Trailing slash ignored; the second URL was never supplied.
        self.assertEqual(diag["evidence_grounded"], 1)

        self.assertEqual(len(state.llm_calls), 1)
        self.assertEqual(state.llm_calls[0]["agent"], "enrichment_agent")

    async def test_no_search_results_skips_gemini(self):

        gemini = FakeGemini([])

        state = make_state([make_lead()])

        agent = make_agent(FakeSearch([[]]), gemini)

        enriched = await agent.run(None, state)

        self.assertEqual(enriched[0].status, "no_search_results")
        self.assertIsNone(enriched[0].details)
        self.assertEqual(gemini.calls, 0)

    async def test_llm_parse_failed(self):

        state = make_state([make_lead()])

        agent = make_agent(
            FakeSearch([RESULTS]),
            FakeGemini([None])
        )

        enriched = await agent.run(None, state)

        self.assertEqual(enriched[0].status, "llm_parse_failed")
        self.assertIsNone(enriched[0].details)

    async def test_llm_error_on_one_lead_does_not_stop_others(self):

        state = make_state([
            make_lead("Alpha"),
            make_lead("Beta")
        ])

        agent = make_agent(
            FakeSearch([RESULTS, RESULTS]),
            FakeGemini([
                FakeAPIError(400, "bad request"),
                make_extract(website="https://beta.example")
            ])
        )

        enriched = await agent.run(None, state)

        self.assertEqual(
            [item.status for item in enriched],
            ["llm_error", "ok"]
        )
        self.assertIn("bad request", enriched[0].error)
        self.assertEqual(
            enriched[1].details.website,
            "https://beta.example"
        )

        diag = state.diagnostics["enrichment_agent"]

        self.assertEqual(
            diag["status_counts"],
            {"llm_error": 1, "ok": 1}
        )

    async def test_search_error_on_one_lead_does_not_stop_others(self):

        state = make_state([
            make_lead("Alpha"),
            make_lead("Beta")
        ])

        gemini = FakeGemini([make_extract()])

        agent = make_agent(
            FakeSearch([RuntimeError("tavily down"), RESULTS]),
            gemini
        )

        enriched = await agent.run(None, state)

        self.assertEqual(
            [item.status for item in enriched],
            ["search_error", "ok"]
        )
        self.assertIn("tavily down", enriched[0].error)
        self.assertEqual(gemini.calls, 1)

    async def test_every_lead_gets_a_result_in_order(self):

        state = make_state([
            make_lead("Alpha"),
            make_lead("Beta"),
            make_lead("Gamma")
        ])

        agent = make_agent(
            FakeSearch([RESULTS, [], RESULTS]),
            FakeGemini([make_extract(), make_extract()])
        )

        enriched = await agent.run(None, state)

        self.assertEqual(
            [item.lead.company_name for item in enriched],
            ["Alpha", "Beta", "Gamma"]
        )
        self.assertEqual(
            [item.status for item in enriched],
            ["ok", "no_search_results", "ok"]
        )

    async def test_requires_leads(self):

        agent = make_agent(
            FakeSearch([]),
            FakeGemini([])
        )

        with self.assertRaises(ValueError):
            await agent.run(None, make_state([]))

    async def test_results_per_lead_is_configurable(self):

        search = FakeSearch([RESULTS])

        agent = make_agent(
            search,
            FakeGemini([make_extract()]),
            results_per_lead=8
        )

        await agent.run(None, make_state([make_lead()]))

        self.assertEqual(search.calls[0][1], 8)

    async def test_default_results_per_lead_matches_original(self):

        search = FakeSearch([RESULTS])

        agent = make_agent(
            search,
            FakeGemini([make_extract()])
        )

        await agent.run(None, make_state([make_lead()]))

        self.assertEqual(search.calls[0][1], 5)


class ProbeEnrichmentTests(unittest.TestCase):

    def test_fill_rates_ignore_failed_enrichments(self):

        enriched = [
            {
                "status": "ok",
                "details": {
                    "website": "https://a.example",
                    "linkedin": None,
                    "founders": ["X"],
                    "evidence": []
                }
            },
            {
                "status": "ok",
                "details": {
                    "website": None,
                    "linkedin": None,
                    "founders": [],
                    "evidence": [{"claim": "c", "url": "u"}]
                }
            },
            {"status": "llm_error", "details": None}
        ]

        rates = fill_rates(enriched)

        self.assertEqual(rates["website"], 0.5)
        self.assertEqual(rates["linkedin"], 0.0)
        self.assertEqual(rates["founders"], 0.5)
        self.assertEqual(rates["evidence"], 0.5)

    def test_fill_rates_empty(self):

        self.assertEqual(fill_rates([]), {})

    def test_summarise(self):

        rows = [
            {
                "startup": "a",
                "leads": 2,
                "status_counts": {"ok": 1, "llm_error": 1},
                "evidence_total": 4,
                "evidence_grounded": 3,
                "enriched": [
                    {"status": "ok", "details": {"website": "w"}},
                    {"status": "llm_error", "details": None}
                ]
            },
            {
                "startup": "b",
                "leads": 0,
                "status_counts": {},
                "evidence_total": 0,
                "evidence_grounded": 0,
                "enriched": []
            }
        ]

        summary = summarise(rows)

        self.assertEqual(summary["startups_without_leads"], 1)
        self.assertEqual(summary["leads_enriched"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["evidence_grounded_rate"], 0.75)
        self.assertEqual(
            summary["status_counts"],
            {"ok": 1, "llm_error": 1}
        )


if __name__ == "__main__":
    unittest.main()