"""
Offline tests for agents/lead_discovery_agent.py diagnostics.

No real API calls: Gemini and Tavily are replaced with fakes.
Run from the project root:

    python -m unittest tests.test_lead_discovery -v
"""

import os
import unittest
from types import SimpleNamespace

# The agent constructor only checks that keys exist.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from agents.lead_discovery_agent import LeadDiscoveryAgent
from core.models import ICP, Lead, LeadList
from core.state import AgentState
from experiments.probe_discovery import summarise


class FakeAPIError(Exception):

    def __init__(self, code, message=""):
        super().__init__(f"{code} {message}")
        self.code = code


class FakeResponse:

    def __init__(self, parsed):
        self.parsed = parsed
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=0
        )


class FakeGemini:
    """Returns (or raises) one scripted outcome per call."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

        self.aio = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=self._generate
            )
        )

    async def _generate(self, model, contents, config):

        self.calls += 1

        if isinstance(self.outcome, Exception):
            raise self.outcome

        return FakeResponse(self.outcome)


class FakeSearch:
    """Returns the same results for every query."""

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    async def async_search(self, query, max_results=5):

        self.calls.append((query, max_results))

        if self.error:
            raise self.error

        return list(self.results)


def make_icp(industries=("Retail", "E-commerce")):

    return ICP(
        target_industries=list(industries),
        target_locations=["India"],
        company_size="SMB"
    )


def make_lead(name, url):

    return Lead(
        company_name=name,
        reason_for_match="matches ICP",
        source_url=url
    )


def make_agent(search, gemini, **kwargs):

    agent = LeadDiscoveryAgent(**kwargs)
    agent.search_tool = search
    agent.client = gemini

    return agent


def make_state(icp=None):

    state = AgentState("test startup")
    state.icp = icp or make_icp()

    return state


RESULTS = [
    {"url": "https://a.example/", "title": "A", "content": "a"},
    {"url": "https://b.example", "title": "B", "content": "b"}
]


class DiscoveryDiagnosticsTests(unittest.IsolatedAsyncioTestCase):

    async def test_ok_counts_and_grounding(self):

        gemini = FakeGemini(LeadList(leads=[
            make_lead("Alpha", "https://a.example"),
            make_lead("Beta", "https://invented.example")
        ]))

        state = make_state()

        agent = make_agent(FakeSearch(RESULTS), gemini)

        leads = await agent.run({}, state)

        diag = state.diagnostics["lead_discovery_agent"]

        self.assertEqual(len(leads), 2)
        self.assertEqual(diag["outcome"], "ok")
        self.assertEqual(diag["search_results_unique"], 2)
        self.assertEqual(diag["leads_from_llm"], 2)
        self.assertEqual(diag["leads_after_dedup"], 2)

        # Alpha's URL was supplied (trailing slash ignored),
        # Beta's was not.
        self.assertEqual(diag["leads_grounded"], 1)

        self.assertEqual(diag["result_titles"], ["A", "B"])

        self.assertEqual(len(state.llm_calls), 1)
        self.assertEqual(state.leads, leads)

    async def test_no_search_results_skips_gemini(self):

        gemini = FakeGemini(LeadList(leads=[]))

        state = make_state()

        agent = make_agent(FakeSearch([]), gemini)

        leads = await agent.run({}, state)

        diag = state.diagnostics["lead_discovery_agent"]

        self.assertEqual(leads, [])
        self.assertEqual(diag["outcome"], "no_search_results")
        self.assertEqual(gemini.calls, 0)

    async def test_llm_returned_empty(self):

        state = make_state()

        agent = make_agent(
            FakeSearch(RESULTS),
            FakeGemini(LeadList(leads=[]))
        )

        leads = await agent.run({}, state)

        self.assertEqual(leads, [])
        self.assertEqual(
            state.diagnostics["lead_discovery_agent"]["outcome"],
            "llm_returned_empty"
        )

    async def test_llm_parse_failed(self):

        state = make_state()

        agent = make_agent(
            FakeSearch(RESULTS),
            FakeGemini(None)
        )

        leads = await agent.run({}, state)

        self.assertEqual(leads, [])
        self.assertEqual(
            state.diagnostics["lead_discovery_agent"]["outcome"],
            "llm_parse_failed"
        )

    async def test_llm_error_is_recorded_and_raised(self):

        state = make_state()

        agent = make_agent(
            FakeSearch(RESULTS),
            FakeGemini(FakeAPIError(400, "bad request"))
        )

        with self.assertRaises(FakeAPIError):
            await agent.run({}, state)

        diag = state.diagnostics["lead_discovery_agent"]

        self.assertEqual(diag["outcome"], "llm_error")
        self.assertIn("bad request", diag["error"])
        self.assertEqual(len(diag["llm_failures"]), 1)
        self.assertEqual(diag["llm_failures"][0]["code"], 400)

    async def test_search_error_is_recorded_and_raised(self):

        state = make_state()

        agent = make_agent(
            FakeSearch(error=RuntimeError("tavily down")),
            FakeGemini(LeadList(leads=[]))
        )

        with self.assertRaises(RuntimeError):
            await agent.run({}, state)

        diag = state.diagnostics["lead_discovery_agent"]

        self.assertEqual(diag["outcome"], "search_error")
        self.assertIn("tavily down", diag["error"])

    async def test_duplicate_company_names_are_merged(self):

        state = make_state()

        agent = make_agent(
            FakeSearch(RESULTS),
            FakeGemini(LeadList(leads=[
                make_lead("Alpha", "https://a.example"),
                make_lead(" alpha ", "https://b.example")
            ]))
        )

        leads = await agent.run({}, state)

        diag = state.diagnostics["lead_discovery_agent"]

        self.assertEqual(len(leads), 1)
        self.assertEqual(diag["leads_from_llm"], 2)
        self.assertEqual(diag["leads_after_dedup"], 1)


class DiscoveryConfigTests(unittest.IsolatedAsyncioTestCase):

    async def test_defaults_match_original_behaviour(self):

        search = FakeSearch(RESULTS)

        agent = make_agent(
            search,
            FakeGemini(LeadList(leads=[]))
        )

        await agent.run({}, make_state())

        # One industry -> 3 queries, 3 results each.
        self.assertEqual(len(search.calls), 3)
        self.assertTrue(all(
            "Retail" in query for query, _ in search.calls
        ))
        self.assertTrue(all(
            max_results == 3 for _, max_results in search.calls
        ))

    async def test_wider_configuration(self):

        search = FakeSearch(RESULTS)

        agent = make_agent(
            search,
            FakeGemini(LeadList(leads=[])),
            max_industries=2,
            results_per_query=5
        )

        state = make_state()

        await agent.run({}, state)

        self.assertEqual(len(search.calls), 6)
        self.assertTrue(all(
            max_results == 5 for _, max_results in search.calls
        ))
        self.assertEqual(
            state.diagnostics["lead_discovery_agent"]["industries"],
            ["Retail", "E-commerce"]
        )

    async def test_requires_icp(self):

        state = AgentState("x")

        agent = make_agent(
            FakeSearch(RESULTS),
            FakeGemini(LeadList(leads=[]))
        )

        with self.assertRaises(ValueError):
            await agent.run({}, state)

    async def test_icp_without_industries_is_rejected(self):

        agent = make_agent(
            FakeSearch(RESULTS),
            FakeGemini(LeadList(leads=[]))
        )

        with self.assertRaises(ValueError):
            await agent.run({}, make_state(make_icp(industries=())))


class ProbeSummaryTests(unittest.TestCase):

    def test_summarise(self):

        rows = [
            {"config": "baseline", "leads": 0,
             "outcome": "llm_returned_empty"},
            {"config": "baseline", "leads": 4, "outcome": "ok"},
            {"config": "wide", "leads": 6, "outcome": "ok"},
            {"config": "wide", "leads": 2, "outcome": "ok"}
        ]

        summary = summarise(rows)

        self.assertEqual(summary["baseline"]["runs"], 2)
        self.assertEqual(summary["baseline"]["zero_lead_runs"], 1)
        self.assertEqual(summary["baseline"]["zero_lead_rate"], 0.5)
        self.assertEqual(summary["baseline"]["mean_leads"], 2.0)
        self.assertEqual(summary["wide"]["zero_lead_rate"], 0.0)
        self.assertEqual(summary["wide"]["outcomes"], {"ok": 2})

    def test_infrastructure_errors_are_separated(self):

        rows = [
            {"config": "baseline", "leads": 0,
             "outcome": "llm_error"},
            {"config": "baseline", "leads": 0,
             "outcome": "llm_returned_empty"},
            {"config": "baseline", "leads": 3, "outcome": "ok"},
            {"config": "baseline", "leads": 2, "outcome": "ok"}
        ]

        summary = summarise(rows)["baseline"]

        # Raw rate counts every zero-lead run...
        self.assertEqual(summary["zero_lead_rate"], 0.5)

        # ...but only 1 of the 3 runs that actually got an
        # answer from Gemini found nothing.
        self.assertEqual(summary["error_runs"], 1)
        self.assertAlmostEqual(
            summary["zero_lead_rate_excluding_errors"],
            0.333,
            places=3
        )

    def test_by_startup_breakdown(self):

        rows = [
            {"startup": "gst", "config": "wide", "leads": 0,
             "outcome": "llm_returned_empty"},
            {"startup": "gst", "config": "wide", "leads": 4,
             "outcome": "ok"},
            {"startup": "gst", "config": "baseline", "leads": 2,
             "outcome": "ok"}
        ]

        by_startup = summarise(rows)["by_startup"]

        self.assertEqual(
            by_startup["gst/wide"],
            {"runs": 2, "zero_lead_runs": 1}
        )
        self.assertEqual(
            by_startup["gst/baseline"],
            {"runs": 1, "zero_lead_runs": 0}
        )


if __name__ == "__main__":
    unittest.main()