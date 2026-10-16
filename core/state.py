from typing import Any


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

    def add_history(self, agent_name: str, result: Any):

        self.history.append({
            "agent": agent_name,
            "result": result
        })

    def add_llm_call(self, agent_name: str, llm_result: Any):

        self.llm_calls.append(
            llm_result.to_record(agent_name)
        )