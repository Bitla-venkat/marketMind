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

    def add_history(self, agent_name: str, result: Any):

        self.history.append({
            "agent": agent_name,
            "result": result
        })