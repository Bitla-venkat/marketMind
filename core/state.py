from typing import Any


# Artifact names that live in an existing AgentState attribute.
# "research" and "outreach" are the existing dict fields.
_ATTRIBUTE_ARTIFACTS = {
    "startup_description": "startup_description",
    "icp": "icp",
    "leads": "leads",
    "enriched_leads": "enriched_leads",
    "scores": "scores",
    "research": "research",
    "outreach": "outreach"
}


class AgentState:

    def __init__(self, startup_description: str):

        self.startup_description = startup_description

        self.icp = None
        self.leads = []
        self.enriched_leads = []
        self.scores = {}
        self.research = {}
        self.outreach = {}

        self.history = []
        self.errors = []

        # One record per Gemini call (model, attempts,
        # latency, tokens). Used for cost/latency metrics.
        self.llm_calls = []

        # Per-agent diagnostic records, keyed by agent name.
        # Explains WHY an agent produced what it did.
        self.diagnostics = {}

        # Artifacts with no dedicated attribute above, by name.
        # New kinds of agents write here, so core code does not
        # need editing for them.
        self.artifacts = {}

    def get_artifact(self, name: str) -> Any:
        """The artifact's value, or None if it was never set."""

        attribute = _ATTRIBUTE_ARTIFACTS.get(name)

        if attribute is not None:
            return getattr(self, attribute)

        return self.artifacts.get(name)

    def set_artifact(self, name: str, value: Any) -> None:

        attribute = _ATTRIBUTE_ARTIFACTS.get(name)

        if attribute is not None:
            setattr(self, attribute, value)
        else:
            self.artifacts[name] = value

    def has_artifact(self, name: str) -> bool:
        """
        True if the artifact exists and is not empty. None, an empty
        list, dict or string all count as "not there", so an agent
        that finished but found nothing does not satisfy the agents
        that need its output.
        """

        value = self.get_artifact(name)

        if value is None:
            return False

        if isinstance(value, (list, dict, str, tuple, set)):
            return len(value) > 0

        return True

    def add_history(self, agent_name: str, result: Any):

        self.history.append({
            "agent": agent_name,
            "result": result
        })

    def add_llm_call(self, agent_name: str, llm_result: Any):

        self.llm_calls.append(
            llm_result.to_record(agent_name)
        )