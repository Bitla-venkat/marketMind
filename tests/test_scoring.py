"""
Offline tests for agents/scoring_agent.py.

No real API calls: Gemini is replaced with a fake.
Run from the project root:

    python -m unittest tests.test_scoring -v
"""

import os
import unittest
from types import SimpleNamespace

# The agent constructor only checks that the key exists.
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from agents.scoring_agent import ScoringAgent
from core.models import (
    EnrichedLead,
    EnrichmentExtract,
    ICP,
    Lead,
    LeadScore,
    LeadScoreList
)
from core.state import AgentState


class FakeAPIError(Exception):

    def __init__(self, code, message=""):
        super().__init__(f"{code} {message}")
        self.code = code


class FakeResponse:

    def __init__(self, parsed):
        self.parsed = parsed
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=300,
            candidates_token_count=60,
            thoughts_token_count=0
        )


class FakeGemini:

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0
        self.prompts = []

        self.aio = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=self._generate
            )
        )

    async def _generate(self, model, contents, config):

        self.calls += 1

        self.prompts.append(contents)

        if isinstance(self.outcome, Exception):
            raise self.outcome

        return FakeResponse(self.outcome)


def make_enriched(name, status="ok"):

    lead = Lead(
        company_name=name,
        reason_for_match="matches ICP"
    )

    return EnrichedLead(
        lead=lead,
        status=status,
        details=(
            EnrichmentExtract(website=f"https://{name}.example")
            if status == "ok" else None
        )
    )


def make_score(name, score):

    return LeadScore(
        company_name=name,
        score=score,
        rationale="because"
    )


def make_state(names=("Alpha", "Beta", "Gamma")):

    state = AgentState("A billing tool for retailers.")

    state.icp = ICP(
        target_industries=["Retail"],
        target_locations=["India"],
        company_size="SMB"
    )

    state.enriched_leads = [make_enriched(n) for n in names]

    return state


def make_agent(gemini):

    agent = ScoringAgent()
    agent.client = gemini

    return agent


class ScoringAgentTests(unittest.IsolatedAsyncioTestCase):

    async def test_ranks_by_score_and_fills_state(self):

        gemini = FakeGemini(LeadScoreList(scores=[
            make_score("Alpha", 40),
            make_score("Beta", 90),
            make_score("Gamma", 65)
        ]))

        state = make_state()

        ranked = await make_agent(gemini).run({}, state)

        self.assertEqual(
            [item.company_name for item in ranked],
            ["Beta", "Gamma", "Alpha"]
        )

        self.assertEqual(
            list(state.scores),
            ["Beta", "Gamma", "Alpha"]
        )
        self.assertEqual(state.scores["Beta"]["score"], 90)

        diag = state.diagnostics["scoring_agent"]

        self.assertEqual(diag["outcome"], "ok")
        self.assertEqual(diag["scored"], 3)
        self.assertEqual(diag["unscored"], [])
        self.assertEqual(diag["score_min"], 40)
        self.assertEqual(diag["score_max"], 90)

        self.assertEqual(gemini.calls, 1)
        self.assertEqual(len(state.llm_calls), 1)
        self.assertEqual(state.llm_calls[0]["agent"], "scoring_agent")

    async def test_prompt_contains_startup_icp_and_leads(self):

        gemini = FakeGemini(LeadScoreList(scores=[]))

        state = make_state(("Alpha",))

        await make_agent(gemini).run({}, state)

        prompt = gemini.prompts[0]

        self.assertIn("A billing tool for retailers.", prompt)
        self.assertIn("Retail", prompt)
        self.assertIn("Alpha", prompt)
        self.assertNotIn("MarketMind", prompt)

    async def test_scores_are_clamped_to_0_100(self):

        gemini = FakeGemini(LeadScoreList(scores=[
            make_score("Alpha", 140),
            make_score("Beta", -5),
            make_score("Gamma", 50)
        ]))

        state = make_state()

        ranked = await make_agent(gemini).run({}, state)

        by_name = {item.company_name: item.score for item in ranked}

        self.assertEqual(by_name["Alpha"], 100)
        self.assertEqual(by_name["Beta"], 0)
        self.assertEqual(by_name["Gamma"], 50)

    async def test_unknown_names_are_dropped(self):

        gemini = FakeGemini(LeadScoreList(scores=[
            make_score("Alpha", 70),
            make_score("Invented Corp", 99)
        ]))

        state = make_state(("Alpha", "Beta"))

        ranked = await make_agent(gemini).run({}, state)

        diag = state.diagnostics["scoring_agent"]

        self.assertEqual(
            [item.company_name for item in ranked],
            ["Alpha"]
        )
        self.assertEqual(diag["unknown_names"], ["Invented Corp"])
        self.assertEqual(diag["unscored"], ["Beta"])
        self.assertNotIn("Invented Corp", state.scores)

    async def test_names_match_ignoring_case_and_spacing(self):

        gemini = FakeGemini(LeadScoreList(scores=[
            make_score("  alpha  ", 70)
        ]))

        state = make_state(("Alpha",))

        ranked = await make_agent(gemini).run({}, state)

        # Reported under the lead's own spelling.
        self.assertEqual(ranked[0].company_name, "Alpha")

    async def test_duplicates_keep_first_entry(self):

        gemini = FakeGemini(LeadScoreList(scores=[
            make_score("Alpha", 70),
            make_score("alpha", 10)
        ]))

        state = make_state(("Alpha",))

        ranked = await make_agent(gemini).run({}, state)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].score, 70)
        self.assertEqual(
            state.diagnostics["scoring_agent"]["duplicates"],
            1
        )

    async def test_no_valid_scores(self):

        gemini = FakeGemini(LeadScoreList(scores=[
            make_score("Invented Corp", 80)
        ]))

        state = make_state(("Alpha",))

        ranked = await make_agent(gemini).run({}, state)

        self.assertEqual(ranked, [])
        self.assertEqual(
            state.diagnostics["scoring_agent"]["outcome"],
            "no_valid_scores"
        )

    async def test_parse_failure(self):

        state = make_state()

        ranked = await make_agent(FakeGemini(None)).run({}, state)

        diag = state.diagnostics["scoring_agent"]

        self.assertEqual(ranked, [])
        self.assertEqual(state.scores, {})
        self.assertEqual(diag["outcome"], "llm_parse_failed")
        self.assertEqual(
            diag["unscored"],
            ["Alpha", "Beta", "Gamma"]
        )

    async def test_llm_error_is_recorded_and_raised(self):

        state = make_state()

        agent = make_agent(
            FakeGemini(FakeAPIError(400, "bad request"))
        )

        with self.assertRaises(FakeAPIError):
            await agent.run({}, state)

        diag = state.diagnostics["scoring_agent"]

        self.assertEqual(diag["outcome"], "llm_error")
        self.assertIn("bad request", diag["error"])
        self.assertEqual(len(diag["llm_failures"]), 1)

    async def test_failed_enrichments_are_still_scored(self):

        gemini = FakeGemini(LeadScoreList(scores=[
            make_score("Alpha", 30),
            make_score("Beta", 60)
        ]))

        state = make_state(("Alpha", "Beta"))
        state.enriched_leads[0] = make_enriched(
            "Alpha",
            status="llm_error"
        )

        ranked = await make_agent(gemini).run({}, state)

        self.assertEqual(len(ranked), 2)
        self.assertIn("llm_error", gemini.prompts[0])

    async def test_requires_icp(self):

        state = make_state()
        state.icp = None

        with self.assertRaises(ValueError):
            await make_agent(FakeGemini(None)).run({}, state)

    async def test_requires_enriched_leads(self):

        state = make_state()
        state.enriched_leads = []

        with self.assertRaises(ValueError):
            await make_agent(FakeGemini(None)).run({}, state)


if __name__ == "__main__":
    unittest.main()
