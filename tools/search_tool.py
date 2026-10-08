import os

from tavily import TavilyClient


class SearchTool:

    name = "web_search"

    description = """
    Searches the web for companies and publicly
    available business information.
    """

    def __init__(self):

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is not set."
            )

        self.client = TavilyClient(
            api_key=api_key
        )

    def search(
        self,
        query: str,
        max_results: int = 5
    ):

        response = self.client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )

        return response.get("results", [])