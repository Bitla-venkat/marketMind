import os
import json

from google import genai

from agents.base import BaseAgent
from core.models import LeadScoreList
from utils.llm_retry import generate_with_fallback


def _key(name: str) -> str:
    """Normalised company name used to match scores to leads."""

    return " ".join(name.strip().lower().split())


class ScoringAgent(BaseAgent):

    name = "scoring_agent"

    description = """
    Scores enriched leads against the Ideal Customer
    Profile and ranks them.
    """

    capabilities = [
        "score_leads",
        "rank_leads",
        "prioritize_leads"
    ]

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    async def run(self, task, state):

        if state.icp is None:
            raise ValueError(
                "ICP must be generated before scoring."
            )

        if not state.enriched_leads:
            raise ValueError(
                "Leads must be enriched before scoring."
            )

        # -----------------------------------------
        # Diagnostics, stored in state.diagnostics.
        #
        # outcome is one of:
        #   ok
        #   no_valid_scores   (Gemini answered, but no entry
        #                      matched a lead we sent)
        #   llm_parse_failed
        #   llm_error
        # -----------------------------------------

        diagnostics = {
            "outcome": None,
            "leads_in": len(state.enriched_leads),
            "scored": 0,
            "unscored": [],
            "unknown_names": [],
            "duplicates": 0,
            "score_min": None,
            "score_max": None,
            "error": None
        }

        state.diagnostics[self.name] = diagnostics

        try:

            return await self._score(state, diagnostics)

        except Exception as e:

            diagnostics["outcome"] = "llm_error"

            diagnostics["error"] = (
                f"{type(e).__name__}: {str(e)[:300]}"
            )

            diagnostics["llm_failures"] = getattr(
                e,
                "llm_failures",
                []
            )

            raise

    async def _score(self, state, diagnostics):

        # -----------------------------------------
        # One compact record per lead. Failed
        # enrichments are included, labelled, so the
        # model knows how much it really knows.
        # -----------------------------------------

        payload = [
            {
                "company_name": enriched.lead.company_name,
                "enrichment_status": enriched.status,
                "discovery": enriched.lead.model_dump(),
                "enrichment": (
                    enriched.details.model_dump()
                    if enriched.details else None
                )
            }
            for enriched in state.enriched_leads
        ]

        prompt = f"""
You are a B2B lead scoring agent.

We are looking for customers for the following
startup:

{state.startup_description}

Ideal Customer Profile:

{state.icp.model_dump_json(indent=2)}

Leads to score:

{json.dumps(payload, indent=2)}

Score EVERY lead from 0 to 100 for how well it
fits the Ideal Customer Profile.

IMPORTANT:

1. Return exactly one entry per lead, using the
   company_name exactly as given.

2. Use ONLY the information provided for each
   lead. Do NOT invent facts.

3. If important information is missing (for
   example enrichment failed), list it in
   missing_information instead of guessing.

4. Use the full range of scores. Give low scores
   to leads that clearly do not fit, and reserve
   high scores for strong fits.

5. In matched_criteria, list the specific ICP
   criteria the lead satisfies (industry,
   location, size, buyer roles, buying signals).

6. In rationale, give one or two sentences.
"""

        llm_result = await generate_with_fallback(
            client=self.client,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": LeadScoreList
            }
        )

        state.add_llm_call(self.name, llm_result)

        parsed = llm_result.response.parsed

        if parsed is None:

            print(
                "Gemini output could not be parsed "
                "into a LeadScoreList."
            )

            diagnostics["outcome"] = "llm_parse_failed"

            diagnostics["unscored"] = [
                enriched.lead.company_name
                for enriched in state.enriched_leads
            ]

            state.scores = {}

            state.add_history(self.name, [])

            return []

        # -----------------------------------------
        # Match returned scores to the leads we sent.
        # Names we never sent are dropped (possible
        # invention); repeats keep the first entry.
        # -----------------------------------------

        known = {
            _key(enriched.lead.company_name):
                enriched.lead.company_name
            for enriched in state.enriched_leads
        }

        scores = {}

        for item in parsed.scores:

            key = _key(item.company_name)

            if key not in known:

                diagnostics["unknown_names"].append(
                    item.company_name
                )

                continue

            if key in scores:

                diagnostics["duplicates"] += 1

                continue

            # Use the lead's own spelling and keep the
            # score in range.
            item.company_name = known[key]

            item.score = max(0, min(100, item.score))

            scores[key] = item

        diagnostics["unscored"] = [
            name
            for key, name in known.items()
            if key not in scores
        ]

        ranked = sorted(
            scores.values(),
            key=lambda item: item.score,
            reverse=True
        )

        diagnostics["scored"] = len(ranked)

        if ranked:
            diagnostics["score_min"] = ranked[-1].score
            diagnostics["score_max"] = ranked[0].score

        diagnostics["outcome"] = (
            "ok" if ranked else "no_valid_scores"
        )

        # -----------------------------------------
        # Save results in shared state
        # -----------------------------------------

        state.scores = {
            item.company_name: item.model_dump()
            for item in ranked
        }

        state.add_history(
            self.name,
            [item.model_dump() for item in ranked]
        )

        print(
            f"\nScored {len(ranked)} of "
            f"{diagnostics['leads_in']} leads"
        )

        return ranked
