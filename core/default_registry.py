"""
The registry holding the agents that exist today.

Agents are imported inside the factories so that importing this
module does not need API keys until an agent is actually built.
"""

from core.registry import AgentRegistry, AgentSpec
from core.validation import (
    make_enrichment_validator,
    make_leads_validator,
    validate_icp,
    validate_scores
)


def _icp_agent(**params):

    from agents.icp_agent import ICPAgent

    return ICPAgent(**params)


def _lead_discovery_agent(**params):

    from agents.lead_discovery_agent import LeadDiscoveryAgent

    return LeadDiscoveryAgent(**params)


def _enrichment_agent(**params):

    from agents.enrichment_agent import EnrichmentAgent

    return EnrichmentAgent(**params)


def _scoring_agent(**params):

    from agents.scoring_agent import ScoringAgent

    return ScoringAgent(**params)


def build_default_registry(
    min_leads: int = 3,
    min_enriched_ok: int = 1
) -> AgentRegistry:
    """
    min_leads: discovery must return at least this many leads.
    min_enriched_ok: enrichment must succeed for at least this many.
    """

    registry = AgentRegistry()

    registry.register(AgentSpec(
        name="icp_agent",
        description="Defines the Ideal Customer Profile.",
        requires=("startup_description",),
        produces=("icp",),
        factory=_icp_agent,
        validator=validate_icp,
        escalations=({},),
        cost_hint={"llm_calls": 1, "searches": 0}
    ))

    registry.register(AgentSpec(
        name="lead_discovery_agent",
        description="Finds candidate companies matching the ICP.",
        requires=("startup_description", "icp"),
        produces=("leads",),
        factory=_lead_discovery_agent,
        validator=make_leads_validator(min_leads),
        # Same settings once (results vary run to run), then the
        # wide variant used by the fixed "wide" baseline.
        escalations=(
            {},
            {"max_industries": 3, "results_per_query": 5}
        ),
        default_params={
            "max_industries": 1,
            "results_per_query": 3
        },
        cost_hint={"llm_calls": 1, "searches": 3}
    ))

    registry.register(AgentSpec(
        name="enrichment_agent",
        description="Gathers evidence and details for each lead.",
        requires=("leads",),
        produces=("enriched_leads",),
        factory=_enrichment_agent,
        validator=make_enrichment_validator(min_enriched_ok),
        escalations=({},),
        default_params={"results_per_lead": 5},
        cost_hint={"llm_calls_per_lead": 1, "searches_per_lead": 1}
    ))

    registry.register(AgentSpec(
        name="scoring_agent",
        description="Scores and ranks enriched leads against the ICP.",
        requires=("startup_description", "icp", "enriched_leads"),
        produces=("scores",),
        factory=_scoring_agent,
        validator=validate_scores,
        escalations=({},),
        cost_hint={"llm_calls": 1, "searches": 0}
    ))

    return registry
