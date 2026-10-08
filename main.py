import asyncio
import json

from dotenv import load_dotenv

from agents.icp_agent import ICPAgent
from agents.lead_discovery_agent import LeadDiscoveryAgent
from agents.enrichment_agent import EnrichmentAgent
from core.state import AgentState


async def main():

    load_dotenv()

    startup_description = """
    MarketMind is an AI marketing platform for Indian
    small and medium-sized businesses.

    It helps businesses generate personalized marketing
    campaigns, identify potential customers, and automate
    parts of their marketing workflow.

    The target market is currently India.
    """

    state = AgentState(
        startup_description
    )

    agent = ICPAgent()

    result = await agent.run(
        task={
            "type": "define_icp"
        },
        state=state
    )

    print("\n===== ICP =====\n")

    print(
        result.model_dump_json(
            indent=2
        )
    )

    lead_agent = LeadDiscoveryAgent()

    leads = await lead_agent.run(
        task={
            "type": "discover_leads"
        },
        state=state
    )
    if not leads:
        print("\nNo leads found. Skipping enrichment.")
        return

    print("\n===== LEADS =====\n")

    for lead in leads:
        print(
            lead.model_dump_json(indent=2)
        )
    enrichment_agent = EnrichmentAgent()

    enriched = await enrichment_agent.run(
        task=None,
        state=state
    )

    print("\n===== ENRICHED LEADS =====")

    for lead in enriched:
        print(json.dumps(
            lead,
            indent=2
        ))


if __name__ == "__main__":
    asyncio.run(main())