"""
Offline tests for tools/search_tool.py.

Tavily is replaced with a fake, so no key or network is needed.
Run from the project root:

    python -m unittest tests.test_search_tool -v
"""

import asyncio
import os
import unittest

os.environ.setdefault("TAVILY_API_KEY", "test-key")

from tools.search_tool import SearchTool


class FakeTavily:

    def __init__(self, fail=False):
        self.fail = fail

    def search(self, query, max_results, search_depth):

        if self.fail:
            raise RuntimeError("tavily down")

        return {"results": [{"url": f"https://{query}.example"}]}


def make_tool(fail=False):

    tool = SearchTool()
    tool.client = FakeTavily(fail=fail)

    return tool


class SearchToolTests(unittest.IsolatedAsyncioTestCase):

    async def test_counts_sync_and_async_searches(self):

        tool = make_tool()

        self.assertEqual(tool.search_count, 0)

        tool.search("a")
        tool.search("b")
        await tool.async_search("c")

        self.assertEqual(tool.search_count, 3)

    async def test_counts_failed_searches_too(self):

        tool = make_tool(fail=True)

        with self.assertRaises(RuntimeError):
            tool.search("a")

        self.assertEqual(tool.search_count, 1)

    async def test_count_is_exact_under_concurrency(self):

        tool = make_tool()

        await asyncio.gather(*[
            tool.async_search(f"q{i}") for i in range(50)
        ])

        self.assertEqual(tool.search_count, 50)

    async def test_results_are_returned(self):

        tool = make_tool()

        results = await tool.async_search("acme")

        self.assertEqual(results[0]["url"], "https://acme.example")


if __name__ == "__main__":
    unittest.main()
