"""
Offline tests for utils/llm_retry.py.

No API key, network access or extra packages needed.
Run from the project root:

    python -m unittest tests.test_llm_retry -v
"""

import asyncio
import time
import unittest
from types import SimpleNamespace

import httpx

from core.state import AgentState
from utils.llm_retry import LLMResult, generate_with_fallback


class FakeResponse:

    def __init__(self):
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=2
        )


class FakeAPIError(Exception):
    """Mimics google.genai errors, which expose .code."""

    def __init__(self, code, message=""):
        super().__init__(f"{code} {message}")
        self.code = code


class FakeClient:
    """
    Replays a script of outcomes, one per API call.

    Each script item is one of:
      - an Exception instance: raised
      - a FakeResponse: returned
      - ("sleep", seconds, FakeResponse): wait, then return

    The blocking path (client.models) fails the test if used,
    proving the code goes through client.aio.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

        self.models = SimpleNamespace(
            generate_content=self._blocking
        )

        self.aio = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=self._async
            )
        )

    def _blocking(self, **kwargs):
        raise AssertionError(
            "Blocking client.models.generate_content was used."
        )

    async def _async(self, model, contents, config):

        self.calls.append(model)

        item = self.script.pop(0)

        if isinstance(item, Exception):
            raise item

        if isinstance(item, tuple):
            _, seconds, response = item
            await asyncio.sleep(seconds)
            return response

        return item


MODELS = ["model-a", "model-b"]


async def call(client, **kwargs):

    return await generate_with_fallback(
        client=client,
        contents="hi",
        config={},
        models=MODELS,
        base_delay=0,
        **kwargs
    )


class GenerateWithFallbackTests(unittest.IsolatedAsyncioTestCase):

    async def test_success_first_try(self):

        client = FakeClient([FakeResponse()])

        result = await call(client)

        self.assertIsInstance(result, LLMResult)
        self.assertEqual(result.model, "model-a")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.thinking_tokens, 2)
        self.assertGreaterEqual(result.latency_s, 0)

    async def test_503_retries_same_model(self):

        client = FakeClient([
            FakeAPIError(503, "overloaded"),
            FakeResponse()
        ])

        result = await call(client)

        self.assertEqual(client.calls, ["model-a", "model-a"])
        self.assertEqual(result.model, "model-a")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0]["code"], 503)

    async def test_429_skips_to_next_model_without_retry(self):

        client = FakeClient([
            FakeAPIError(429, "quota"),
            FakeResponse()
        ])

        result = await call(client)

        self.assertEqual(client.calls, ["model-a", "model-b"])
        self.assertEqual(result.model, "model-b")
        self.assertEqual(result.attempts, 2)

    async def test_repeated_503_falls_back_to_next_model(self):

        client = FakeClient([
            FakeAPIError(503),
            FakeAPIError(503),
            FakeResponse()
        ])

        result = await call(client)

        self.assertEqual(
            client.calls,
            ["model-a", "model-a", "model-b"]
        )
        self.assertEqual(result.model, "model-b")
        self.assertEqual(result.attempts, 3)

    async def test_non_temporary_error_raises_immediately(self):

        client = FakeClient([
            FakeAPIError(400, "bad request"),
            FakeResponse()
        ])

        with self.assertRaises(FakeAPIError):
            await call(client)

        self.assertEqual(client.calls, ["model-a"])

    async def test_all_models_fail_raises_last_error(self):

        client = FakeClient([
            FakeAPIError(429, "a"),
            FakeAPIError(429, "b")
        ])

        with self.assertRaises(FakeAPIError) as ctx:
            await call(client)

        self.assertIn("b", str(ctx.exception))
        self.assertEqual(client.calls, ["model-a", "model-b"])

    async def test_error_without_code_uses_message_fallback(self):

        client = FakeClient([
            Exception("503 Service Unavailable"),
            FakeResponse()
        ])

        result = await call(client)

        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.model, "model-a")

    async def test_timeout_is_retried(self):

        client = FakeClient([
            ("sleep", 1.0, FakeResponse()),
            FakeResponse()
        ])

        result = await call(client, timeout_s=0.05)

        self.assertEqual(client.calls, ["model-a", "model-a"])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.failures[0]["code"], 504)

    async def test_dropped_connection_is_retried_same_model(self):

        # Seen in a real run: httpx.ReadError, which has no
        # HTTP status code.
        client = FakeClient([
            httpx.ReadError(""),
            FakeResponse()
        ])

        result = await call(client)

        self.assertEqual(client.calls, ["model-a", "model-a"])
        self.assertEqual(result.model, "model-a")
        self.assertEqual(result.attempts, 2)
        self.assertIsNone(result.failures[0]["code"])
        self.assertEqual(result.failures[0]["type"], "ReadError")

    async def test_server_disconnected_falls_back_to_next_model(self):

        # Seen in a real run: RemoteProtocolError,
        # "Server disconnected without sending a response."
        client = FakeClient([
            httpx.RemoteProtocolError("Server disconnected"),
            httpx.RemoteProtocolError("Server disconnected"),
            FakeResponse()
        ])

        result = await call(client)

        self.assertEqual(
            client.calls,
            ["model-a", "model-a", "model-b"]
        )
        self.assertEqual(result.model, "model-b")
        self.assertEqual(result.attempts, 3)

    async def test_builtin_connection_error_is_retried(self):

        client = FakeClient([
            ConnectionResetError("reset by peer"),
            FakeResponse()
        ])

        result = await call(client)

        self.assertEqual(result.attempts, 2)

    async def test_unrelated_error_is_still_fatal(self):

        client = FakeClient([
            ValueError("programming error"),
            FakeResponse()
        ])

        with self.assertRaises(ValueError):
            await call(client)

        self.assertEqual(client.calls, ["model-a"])

    async def test_failure_history_attached_to_raised_error(self):

        client = FakeClient([
            FakeAPIError(429, "a"),
            FakeAPIError(429, "b")
        ])

        with self.assertRaises(FakeAPIError) as ctx:
            await call(client)

        failures = ctx.exception.llm_failures

        self.assertEqual(len(failures), 2)
        self.assertEqual(failures[0]["model"], "model-a")
        self.assertEqual(failures[1]["model"], "model-b")

    async def test_calls_run_concurrently(self):

        # Two 0.3s calls. If the wrapper blocked the event
        # loop they would take ~0.6s; concurrently ~0.3s.
        client_1 = FakeClient([("sleep", 0.3, FakeResponse())])
        client_2 = FakeClient([("sleep", 0.3, FakeResponse())])

        started = time.perf_counter()

        await asyncio.gather(
            call(client_1),
            call(client_2)
        )

        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.5)


class StateRecordingTests(unittest.IsolatedAsyncioTestCase):

    async def test_add_llm_call_stores_record(self):

        client = FakeClient([FakeResponse()])
        result = await call(client)

        state = AgentState("test startup")
        state.add_llm_call("icp_agent", result)

        self.assertEqual(len(state.llm_calls), 1)

        record = state.llm_calls[0]

        self.assertEqual(record["agent"], "icp_agent")
        self.assertEqual(record["model"], "model-a")
        self.assertEqual(record["input_tokens"], 10)


if __name__ == "__main__":
    unittest.main()