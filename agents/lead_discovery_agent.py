import os
import json

from google import genai

from agents.base import BaseAgent
from core.models import LeadList
from utils.llm_retry import generate_with_fallback
from tools.search_tool import SearchTool


class LeadDiscoveryAgent(BaseAgent):

    name = "lead_discovery_agent"

    description = """
    Finds potential customers that match an
    Ideal Customer Profile.
    """

    capabilities = [
        "find_companies",
        "find_potential_customers",
        "lead_discovery"
    ]

    def __init__(
        self,
        max_industries: int = 1,
        results_per_query: int = 3
    ):
        """
        max_industries:
            How many of the ICP's target industries to search.
            The default (1) matches the original test-mode
            behaviour.

        results_per_query:
            Tavily results requested per query. The default (3)
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

        self.max_industries = max_industries
        self.results_per_query = results_per_query

    async def run(self, task, state):

        if state.icp is None:
            raise ValueError(
                "ICP must be generated before "
                "lead discovery."
            )

        icp = state.icp

        if not icp.target_industries:
            raise ValueError(
                "ICP has no target industries."
            )

        industries = icp.target_industries[
            :self.max_industries
        ]

        # -----------------------------------------
        # Diagnostics: records WHERE a run succeeds
        # or fails, stored in state.diagnostics.
        #
        # outcome is one of:
        #   ok
        #   no_search_results
        #   llm_parse_failed
        #   llm_returned_empty
        #   search_error
        #   llm_error
        # -----------------------------------------

        diagnostics = {
            "outcome": None,
            "stage": "search",
            "industries": industries,
            "results_per_query": self.results_per_query,
            "queries": [],
            "search_results_total": 0,
            "search_results_unique": 0,
            "leads_from_llm": 0,
            "leads_after_dedup": 0,
            "leads_grounded": 0,
            "error": None
        }

        state.diagnostics[self.name] = diagnostics

        try:

            return await self._discover(
                state,
                icp,
                industries,
                diagnostics
            )

        except Exception as e:

            diagnostics["outcome"] = (
                f"{diagnostics['stage']}_error"
            )

            diagnostics["error"] = (
                f"{type(e).__name__}: {str(e)[:300]}"
            )

            # Failed Gemini attempts, if the error came from
            # generate_with_fallback().
            diagnostics["llm_failures"] = getattr(
                e,
                "llm_failures",
                []
            )

            raise

    async def _discover(
        self,
        state,
        icp,
        industries,
        diagnostics
    ):

        # -----------------------------------------
        # Generate targeted queries
        # -----------------------------------------

        queries = []

        for industry in industries:

            queries.extend([
                f"{industry} startups India",

                f"{industry} companies "
                f"Bengaluru Mumbai Hyderabad",

                f"{industry} companies "
                f"Delhi NCR Pune Chennai"
            ])

        all_search_results = []

        # -----------------------------------------
        # Tavily searches
        # -----------------------------------------

        for query in queries:

            print(f"\nSearching: {query}")

            search_results = await self.search_tool.async_search(
                query=query,
                max_results=self.results_per_query
            )

            diagnostics["queries"].append({
                "query": query,
                "results": len(search_results)
            })

            if not search_results:
                print(
                    "No results found for this query."
                )
                continue

            all_search_results.extend(
                search_results
            )

        diagnostics["search_results_total"] = len(
            all_search_results
        )

        # -----------------------------------------
        # Remove duplicate search results
        # -----------------------------------------

        unique_results = []

        seen_urls = set()

        for result in all_search_results:

            url = result.get("url")

            if url and url not in seen_urls:

                seen_urls.add(url)

                unique_results.append(result)

        diagnostics["search_results_unique"] = len(
            unique_results
        )

        # Titles let you inspect what the model was shown
        # when a run returns no leads.
        diagnostics["result_titles"] = [
            result.get("title")
            for result in unique_results
        ]

        if not unique_results:

            print(
                "No search results found."
            )

            diagnostics["outcome"] = "no_search_results"

            state.leads = []

            state.add_history(self.name, [])

            return []

        print(
            f"\nCollected "
            f"{len(unique_results)} unique search results."
        )

        # -----------------------------------------
        # Prepare results for Gemini
        # -----------------------------------------

        results_text = json.dumps(
            unique_results,
            indent=2
        )

        industry_text = ", ".join(industries)

        # -----------------------------------------
        # Gemini qualification prompt
        # -----------------------------------------

        prompt = f"""
You are a B2B lead qualification agent.

We are looking for potential customers for
MarketMind, an AI marketing platform for Indian
small and medium-sized businesses.

Ideal Customer Profile:

{icp.model_dump_json(indent=2)}

Target industry for this search:

{industry_text}

Here are web search results:

{results_text}

Your task is to identify ALL companies in these
search results that reasonably match the ICP.

For EVERY suitable company:

- Extract the company name.
- Extract the website if available.
- Identify the industry.
- Identify the location.
- Identify company size if available.
- Explain why the company matches the ICP.
- Include the source URL.

IMPORTANT:

1. Return MULTIPLE leads if multiple companies
match the ICP.

2. Do NOT stop after finding the first company.

3. Do NOT invent information.

4. Only use information supported by the
supplied search results.

5. If company size is unknown, do not guess it.

6. If location is unknown, do not guess it.

7. If the same company appears in multiple
search results, return it only once.

8. Exclude companies that clearly do not match
the ICP.

9. Return an empty leads list if no suitable
companies are found.
"""

        # -----------------------------------------
        # Gemini request with model fallback
        # -----------------------------------------

        diagnostics["stage"] = "llm"

        llm_result = await generate_with_fallback(
            client=self.client,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": LeadList
            }
        )

        state.add_llm_call(self.name, llm_result)

        parsed = llm_result.response.parsed

        # -----------------------------------------
        # Process Gemini response
        # -----------------------------------------

        if parsed is None:

            print(
                "Gemini output could not be parsed "
                "into a LeadList."
            )

            diagnostics["outcome"] = "llm_parse_failed"

            state.leads = []

            state.add_history(self.name, [])

            return []

        leads = list(parsed.leads)

        diagnostics["leads_from_llm"] = len(leads)

        for lead in leads:

            print(
                f"\nLead found: "
                f"{lead.company_name}"
            )

        # -----------------------------------------
        # Remove duplicate companies
        # -----------------------------------------

        unique_leads = []

        seen_companies = set()

        for lead in leads:

            company_key = (
                lead.company_name
                .strip()
                .lower()
            )

            if company_key not in seen_companies:

                seen_companies.add(
                    company_key
                )

                unique_leads.append(
                    lead
                )

        diagnostics["leads_after_dedup"] = len(
            unique_leads
        )

        # -----------------------------------------
        # Grounding check: does each lead's source URL
        # appear in the search results we supplied?
        # A lead whose URL we never supplied may have
        # been invented by the model.
        # -----------------------------------------

        known_urls = {
            url.rstrip("/")
            for url in seen_urls
        }

        diagnostics["leads_grounded"] = sum(
            1
            for lead in unique_leads
            if lead.source_url
            and lead.source_url.rstrip("/") in known_urls
        )

        if unique_leads:
            diagnostics["outcome"] = "ok"
        else:
            diagnostics["outcome"] = "llm_returned_empty"

        # -----------------------------------------
        # Save results in shared state
        # -----------------------------------------

        state.leads = unique_leads

        state.add_history(
            self.name,
            [
                lead.model_dump()
                for lead in unique_leads
            ]
        )

        print(
            f"\nTotal unique leads found: "
            f"{len(unique_leads)} "
            f"(outcome: {diagnostics['outcome']})"
        )

        return unique_leads