"""
Offline tests for core/validation.py.

Run from the project root:

    python -m unittest tests.test_validation -v
"""

import os
import unittest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from core.models import ICP, EnrichedLead, Lead
from core.state import AgentState
from core.validation import (
    make_enrichment_validator,
    make_leads_validator,
    validate_icp,
    validate_scores
)


def lead(name="A"):

    return Lead(company_name=name, reason_for_match="fits")


class IcpTests(unittest.TestCase):

    def test_missing_icp_fails(self):

        result = validate_icp(AgentState("s"))

        self.assertFalse(result.passed)

    def test_icp_without_industries_fails(self):

        state = AgentState("s")
        state.icp = ICP(
            target_industries=[],
            target_locations=["India"],
            company_size="SMB"
        )

        self.assertFalse(validate_icp(state).passed)

    def test_good_icp_passes(self):

        state = AgentState("s")
        state.icp = ICP(
            target_industries=["Retail"],
            target_locations=["India"],
            company_size="SMB"
        )

        result = validate_icp(state)

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["industries"], 1)


class LeadsTests(unittest.TestCase):

    def test_threshold(self):

        validator = make_leads_validator(3)

        state = AgentState("s")

        for count, expected in [(0, False), (2, False),
                                (3, True), (5, True)]:

            state.leads = [lead(str(i)) for i in range(count)]

            self.assertEqual(
                validator(state).passed,
                expected,
                count
            )

    def test_reason_and_metrics(self):

        state = AgentState("s")
        state.leads = [lead()]

        result = make_leads_validator(3)(state)

        self.assertIn("Found 1", result.reason)
        self.assertEqual(result.metrics["leads"], 1)
        self.assertEqual(result.metrics["min_leads"], 3)


class EnrichmentTests(unittest.TestCase):

    def test_counts_only_ok_leads(self):

        state = AgentState("s")
        state.enriched_leads = [
            EnrichedLead(lead=lead("A"), status="llm_error"),
            EnrichedLead(lead=lead("B"), status="no_search_results")
        ]

        result = make_enrichment_validator(1)(state)

        self.assertFalse(result.passed)
        self.assertEqual(result.metrics["enriched_ok"], 0)
        self.assertEqual(result.metrics["enriched_total"], 2)

        state.enriched_leads.append(
            EnrichedLead(lead=lead("C"), status="ok")
        )

        self.assertTrue(make_enrichment_validator(1)(state).passed)
        self.assertFalse(make_enrichment_validator(2)(state).passed)


class ScoresTests(unittest.TestCase):

    def test_empty_scores_fail(self):

        self.assertFalse(validate_scores(AgentState("s")).passed)

    def test_out_of_range_score_fails(self):

        state = AgentState("s")
        state.scores = {"A": {"score": 140}}

        self.assertFalse(validate_scores(state).passed)

    def test_non_numeric_score_fails(self):

        state = AgentState("s")
        state.scores = {"A": {"score": None}}

        self.assertFalse(validate_scores(state).passed)

    def test_good_scores_pass(self):

        state = AgentState("s")
        state.scores = {
            "A": {"score": 80},
            "B": {"score": 0}
        }

        result = validate_scores(state)

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["scored"], 2)

    def test_to_dict_is_json_friendly(self):

        import json

        json.dumps(validate_scores(AgentState("s")).to_dict())


if __name__ == "__main__":
    unittest.main()
