"""
Offline tests for experiments/run_baseline.py (summary logic only).

Run from the project root:

    python -m unittest tests.test_run_baseline -v
"""

import os
import unittest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from experiments.run_baseline import summarise_runs


def make_record(
    pipeline="fixed-narrow",
    success=True,
    failed_step=None,
    leads_scored=3,
    latency_s=10.0,
    llm_calls=6,
    search_calls=5
):

    return {
        "pipeline": pipeline,
        "success": success,
        "failed_step": failed_step,
        "leads_scored": leads_scored,
        "latency_s": latency_s,
        "llm_calls": llm_calls,
        "llm_failed_attempts": 0,
        "input_tokens": 1000,
        "output_tokens": 100,
        "search_calls": search_calls
    }


class SummariseRunsTests(unittest.TestCase):

    def test_averages_and_success_rate(self):

        records = [
            make_record(latency_s=10, llm_calls=6),
            make_record(latency_s=20, llm_calls=8),
            make_record(
                success=False,
                failed_step="enrichment_agent",
                leads_scored=0,
                latency_s=30,
                llm_calls=4
            ),
            make_record(
                success=False,
                leads_scored=0,
                latency_s=20,
                llm_calls=6
            )
        ]

        summary = summarise_runs(records)["fixed-narrow"]

        self.assertEqual(summary["runs"], 4)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(
            summary["failed_steps"],
            {"enrichment_agent": 1}
        )

        # Raised nothing but scored nothing: the 4th record.
        self.assertEqual(summary["finished_with_no_scored_leads"], 1)

        self.assertEqual(summary["mean_latency_s"], 20.0)
        self.assertEqual(summary["mean_llm_calls"], 6.0)
        self.assertEqual(summary["mean_leads_scored"], 1.5)

    def test_pipelines_are_summarised_separately(self):

        records = [
            make_record(pipeline="fixed-narrow"),
            make_record(pipeline="fixed-wide", success=False,
                        failed_step="lead_discovery_agent",
                        leads_scored=0)
        ]

        summary = summarise_runs(records)

        self.assertEqual(
            sorted(summary),
            ["fixed-narrow", "fixed-wide"]
        )
        self.assertEqual(summary["fixed-narrow"]["success_rate"], 1.0)
        self.assertEqual(summary["fixed-wide"]["success_rate"], 0.0)

    def test_empty_input(self):

        self.assertEqual(summarise_runs([]), {})


if __name__ == "__main__":
    unittest.main()
