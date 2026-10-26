from pydantic import BaseModel, Field


class ICP(BaseModel):
    target_industries: list[str] = Field(default_factory=list)
    target_locations: list[str] = Field(default_factory=list)

    company_size: str

    buyer_roles: list[str] = Field(default_factory=list)

    pain_points: list[str] = Field(default_factory=list)

    buying_signals: list[str] = Field(default_factory=list)

    exclusions: list[str] = Field(default_factory=list)


class Lead(BaseModel):

    company_name: str

    website: str | None = None

    industry: str | None = None

    location: str | None = None

    company_size: str | None = None

    reason_for_match: str

    source_url: str | None = None


class LeadList(BaseModel):
    leads: list[Lead]


class EvidenceItem(BaseModel):
    """One claim about a company and the page that supports it."""

    claim: str

    url: str


class EnrichmentExtract(BaseModel):
    """
    What Gemini extracts about one company. This is the
    response schema for the enrichment call.
    """

    website: str | None = None

    linkedin: str | None = None

    company_size: str | None = None

    location: str | None = None

    industry: str | None = None

    founders: list[str] = Field(default_factory=list)

    buyer_roles: list[str] = Field(default_factory=list)

    products_services: list[str] = Field(default_factory=list)

    buying_signals: list[str] = Field(default_factory=list)

    evidence: list[EvidenceItem] = Field(default_factory=list)


class EnrichedLead(BaseModel):
    """
    Result of enriching one lead. Always has the same shape,
    whether enrichment worked or not.

    status is one of:
      ok
      no_search_results
      llm_parse_failed
      search_error
      llm_error

    details is None unless status is "ok".
    """

    lead: Lead

    status: str

    error: str | None = None

    details: EnrichmentExtract | None = None


class LeadScore(BaseModel):
    """
    How well one company fits the ICP, as judged by Gemini.

    score is meant to be 0-100. It is not constrained in the
    response schema; the scoring agent clamps it.
    """

    company_name: str

    score: int

    rationale: str

    matched_criteria: list[str] = Field(default_factory=list)

    missing_information: list[str] = Field(default_factory=list)


class LeadScoreList(BaseModel):
    scores: list[LeadScore]