import os
import json

from google import genai

from agents.base import BaseAgent
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

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.search_tool = SearchTool()

    async def run(self, task, state):

        if not state.leads:
            raise ValueError(
                "Leads must be discovered before "
                "enrichment."
            )

        enriched_leads = []

        # -----------------------------------------
        # Enrich every discovered lead
        # -----------------------------------------

        for lead in state.leads:

            print(
                f"\nEnriching: "
                f"{lead.company_name}"
            )

            query = (
                f'"{lead.company_name}" '
                f'India company website '
                f'founder employees LinkedIn '
                f'marketing'
            )

            print(
                f"Searching: {query}"
            )

            search_results = self.search_tool.search(
                query=query,
                max_results=5
            )

            if not search_results:

                print(
                    "No enrichment results found."
                )

                enriched_leads.append(
                    lead.model_dump()
                )

                continue

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

4. Include evidence URLs for important claims.

5. Keep the original company identity intact.

Return a JSON object with:

company_name
website
linkedin
company_size
location
industry
founders
buyer_roles
products_services
buying_signals
evidence
"""

            response = await generate_with_fallback(
                client=self.client,
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            if response is None:

                print(
                    f"Enrichment failed for "
                    f"{lead.company_name}"
                )

                enriched_leads.append(
                    lead.model_dump()
                )

                continue

            try:

                enriched_data = json.loads(
                    response.text
                )

                enriched_leads.append(
                    enriched_data
                )

                print(
                    f"Enriched: "
                    f"{lead.company_name}"
                )

            except Exception as e:

                print(
                    f"Could not parse enrichment "
                    f"for {lead.company_name}: {e}"
                )

                enriched_leads.append(
                    lead.model_dump()
                )

        # -----------------------------------------
        # Save enriched data
        # -----------------------------------------

        state.enriched_leads = enriched_leads

        state.add_history(
            self.name,
            enriched_leads
        )

        print(
            f"\nTotal enriched leads: "
            f"{len(enriched_leads)}"
        )

        return enriched_leads