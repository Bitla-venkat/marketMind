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