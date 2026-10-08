from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):

    name: str = ""
    description: str = ""
    capabilities: list[str] = []

    @abstractmethod
    async def run(self, task: Any, state: Any) -> Any:
        """
        Execute the agent's task.

        task:
            Information about what the agent needs to do.

        state:
            Shared information available to the agent.
        """
        pass