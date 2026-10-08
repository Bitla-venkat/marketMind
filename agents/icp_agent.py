import os

from google import genai

from agents.base import BaseAgent
from core.models import ICP
from utils.llm_retry import generate_with_fallback


class ICPAgent(BaseAgent):

    name = "icp_agent"

    description = """
    Defines the Ideal Customer Profile for a startup.
    """

    capabilities = [
        "define_icp",
        "customer_segmentation",
        "identify_buyer_roles"
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

    async def run(self, task, state):

        startup_description = state.startup_description

        prompt = f"""
You are an expert B2B customer acquisition strategist.

Analyze the following startup:

{startup_description}

Create an Ideal Customer Profile (ICP).

Identify:

1. Target industries
2. Target geographic locations
3. Ideal company size
4. Buyer roles
5. Main pain points
6. Buying signals
7. Companies that should be excluded

Do not invent highly specific facts about the startup.
Base your recommendations on the provided description.
"""

        response = await generate_with_fallback(
            client=self.client,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ICP
            }
        )
        icp = response.parsed

        state.icp = icp
        state.add_history(
            self.name,
            icp.model_dump()
        )

        return icp