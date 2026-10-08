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

            if state.icp is None:
                raise ValueError(
                    "ICP must be generated before "
                    "lead discovery."
                )

            icp = state.icp

            leads = []

            # -----------------------------------------
            # TEST MODE
            # Search only the first industry.
            # -----------------------------------------

            industry = icp.target_industries[0]

            # -----------------------------------------
            # Generate multiple targeted queries
            # -----------------------------------------

            queries = [
                f"{industry} startups India",

                f"{industry} companies "
                f"Bengaluru Mumbai Hyderabad",

                f"{industry} companies "
                f"Delhi NCR Pune Chennai"
            ]

            all_search_results = []

            # -----------------------------------------
            # Tavily searches
            # -----------------------------------------

            for query in queries:

                print(f"\nSearching: {query}")

                search_results = self.search_tool.search(
                    query=query,
                    max_results=3
                )

                if not search_results:
                    print(
                        "No results found for this query."
                    )
                    continue

                all_search_results.extend(
                    search_results
                )

            # -----------------------------------------
            # Check whether we found anything
            # -----------------------------------------

            if not all_search_results:

                print(
                    "No search results found."
                )

                state.leads = []

                return []

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

    {industry}

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

            response = await generate_with_fallback(
                client=self.client,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": LeadList
                }
            )

            # -----------------------------------------
            # Process Gemini response
            # -----------------------------------------

            if response is None:

                print(
                    "Gemini returned no response."
                )

                state.leads = []

                return []

            result = response.parsed

            if result:

                for lead in result.leads:

                    leads.append(lead)

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
                f"{len(unique_leads)}"
            )

            return unique_leads