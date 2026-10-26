import os
import json

from google import genai

from agents.base import BaseAgent
from core.models import EnrichedLead, EnrichmentExtract
from utils.llm_retry import generate_with_fallback
from tools.search_tool import SearchTool


class EnrichmentAgent(BaseAgent):

    name = "enrichment_agent"

    description = """
    Enriches discovered leads with additional
    company information and evidence.
    """

    capabilities = [
        "enrich_company",
        "research_company",
        "find_company_information"
    ]

    def __init__(self, results_per_lead: int = 5):
        """
        results_per_lead:
            Tavily results requested per company. The default (5)
            matches the original behaviour.
        """

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.search_tool = SearchTool()

        self.results_per_lead = results_per_lead

    async def run(self, task, state):

        if not state.leads:
            raise ValueError(
                "Leads must be discovered before "
                "enrichment."
            )

        # -----------------------------------------
        # Diagnostics, stored in state.diagnostics.
        # -----------------------------------------

        diagnostics = {
            "leads_in": len(state.leads),
            "status_counts": {},
            "evidence_total": 0,
            "evidence_grounded": 0,
            "per_lead": []
        }

        state.diagnostics[self.name] = diagnostics

        enriched_leads = []

        # -----------------------------------------
        # Enrich every discovered lead. A failure on
        # one lead is recorded and never stops the
        # others.
        # -----------------------------------------

        for lead in state.leads:

            print(
                f"\nEnriching: "
                f"{lead.company_name}"
            )

            record = {
                "company": lead.company_name,
                "status": None,
                "search_results": 0,
                "evidence": 0,
                "evidence_grounded": 0,
                "error": None
            }

            # Updated by _enrich_one() so a failure can be
            # labelled search_error or llm_error.
            stage_holder = {"stage": "search"}

            try:

                enriched = await self._enrich_one(
                    lead,
                    state,
                    record,
                    stage_holder
                )

            except Exception as e:

                enriched = EnrichedLead(
                    lead=lead,
                    status=f"{stage_holder['stage']}_error",
                    error=(
                        f"{type(e).__name__}: "
                        f"{str(e)[:300]}"
                    )
                )

                print(
                    f"Enrichment failed for "
                    f"{lead.company_name}: {enriched.error}"
                )

            record["status"] = enriched.status
            record["error"] = enriched.error

            counts = diagnostics["status_counts"]

            counts[enriched.status] = (
                counts.get(enriched.status, 0) + 1
            )

            diagnostics["evidence_total"] += record["evidence"]

            diagnostics["evidence_grounded"] += record[
                "evidence_grounded"
            ]

            diagnostics["per_lead"].append(record)

            enriched_leads.append(enriched)

        # -----------------------------------------
        # Save enriched data
        # -----------------------------------------

        state.enriched_leads = enriched_leads

        state.add_history(
            self.name,
            [
                enriched.model_dump()
                for enriched in enriched_leads
            ]
        )

        print(
            f"\nTotal enriched leads: "
            f"{len(enriched_leads)} "
            f"(status: {diagnostics['status_counts']})"
        )

        return enriched_leads

    async def _enrich_one(
        self,
        lead,
        state,
        record,
        stage_holder
    ):
        """
        Enrich a single lead and return an EnrichedLead.

        Raises on search/Gemini errors; run() catches them and
        uses stage_holder["stage"] to say which stage failed.
        """

        query = (
            f'"{lead.company_name}" '
            f'India company website '
            f'founder employees LinkedIn '
            f'marketing'
        )

        print(
            f"Searching: {query}"
        )

        search_results = await self.search_tool.async_search(
            query=query,
            max_results=self.results_per_lead
        )

        record["search_results"] = len(search_results)

        if not search_results:

            print(
                "No enrichment results found."
            )

            return EnrichedLead(
                lead=lead,
                status="no_search_results"
            )

        results_text = json.dumps(
            search_results,
            indent=2
        )

        # -------------------------------------
        # Ask Gemini to extract evidence
        # -------------------------------------

        prompt = f"""
You are a company research and enrichment agent.

We discovered this potential lead:

{lead.model_dump_json(indent=2)}

Search results:

{results_text}

Enrich this company using ONLY information
supported by the search results.

Find, when available:

- Official website
- LinkedIn/company profile
- Company size
- Location
- Industry
- Founders
- Relevant buyer roles
- Recent marketing activity
- Hiring activity
- Product/service information
- Expansion or funding signals
- Other useful buying signals

IMPORTANT:

1. Do NOT invent information.

2. If information cannot be verified,
   return null or an empty list.

3. Prefer information from official company
   websites or reliable company profiles.

4. For important claims, add an evidence item
   with the claim and the URL that supports it.
   The URL MUST be one of the URLs in the search
   results above.

5. The company identity is fixed. Do not
   rename it or switch to a different company
   with a similar name.
"""

        stage_holder["stage"] = "llm"

        llm_result = await generate_with_fallback(
            client=self.client,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": EnrichmentExtract
            }
        )

        state.add_llm_call(self.name, llm_result)

        extract = llm_result.response.parsed

        if extract is None:

            print(
                f"Gemini output could not be parsed for "
                f"{lead.company_name}"
            )

            return EnrichedLead(
                lead=lead,
                status="llm_parse_failed"
            )

        # -------------------------------------
        # Grounding check: does each evidence URL
        # appear in the search results we supplied?
        # -------------------------------------

        known_urls = {
            result["url"].rstrip("/")
            for result in search_results
            if result.get("url")
        }

        record["evidence"] = len(extract.evidence)

        record["evidence_grounded"] = sum(
            1
            for item in extract.evidence
            if item.url.rstrip("/") in known_urls
        )

        print(
            f"Enriched: "
            f"{lead.company_name}"
        )

        return EnrichedLead(
            lead=lead,
            status="ok",
            details=extract
        )