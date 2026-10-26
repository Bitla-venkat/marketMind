import asyncio
import os
import threading

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

        # Number of searches attempted. Read by the pipeline
        # runner to report search cost. Searches may run in
        # worker threads, so updates are guarded by a lock.
        self.search_count = 0
        self._count_lock = threading.Lock()

    def search(
        self,
        query: str,
        max_results: int = 5
    ):

        with self._count_lock:
            self.search_count += 1

        response = self.client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )

        return response.get("results", [])

    async def async_search(
        self,
        query: str,
        max_results: int = 5
    ):
        """
        Non-blocking version of search().

        Runs the blocking Tavily call in a worker thread so
        the asyncio event loop stays free for other tasks.
        """

        return await asyncio.to_thread(
            self.search,
            query=query,
            max_results=max_results
        )