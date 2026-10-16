import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

# httpx is a dependency of google-genai, so it is always installed.
import httpx


DEFAULT_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite"
]

# Errors worth retrying on the SAME model after a short wait.
RETRYABLE_CODES = {500, 502, 503, 504}

# Quota / rate-limit errors. Waiting a few seconds rarely
# helps, so we move straight to the next model.
QUOTA_CODES = {429}


@dataclass
class LLMResult:
    """
    Outcome of one generate_with_fallback() call.

    response:
        The raw Gemini response object.

    model:
        The model that finally produced the response.

    attempts:
        Total API calls made, including failed ones.

    latency_s:
        Wall-clock seconds for the whole call, including
        retries and backoff sleeps.

    input_tokens / output_tokens / thinking_tokens:
        Token usage reported by Gemini for the successful
        call. None if the API did not report it.

    failures:
        One dict per failed attempt (model, attempt, code,
        error), so failures can be counted in experiments.
    """

    response: Any
    model: str
    attempts: int
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    failures: list[dict] = field(default_factory=list)

    def to_record(self, agent_name: str | None = None) -> dict:
        """Plain-dict form, safe to store or write as JSON."""

        return {
            "agent": agent_name,
            "model": self.model,
            "attempts": self.attempts,
            "latency_s": round(self.latency_s, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "failures": self.failures
        }


def _error_code(error: Exception) -> int | None:
    """
    Best-effort HTTP status code for an exception.

    The google-genai SDK exposes .code on its API errors.
    For anything else, fall back to looking for a known
    status code in the message (the previous behaviour).
    """

    code = getattr(error, "code", None)

    if isinstance(code, int):
        return code

    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return 504

    error_text = str(error)

    for candidate in sorted(QUOTA_CODES | RETRYABLE_CODES):
        if str(candidate) in error_text:
            return candidate

    return None


def _is_network_error(error: Exception) -> bool:
    """
    True for connection-level failures that carry no HTTP
    status code, such as httpx.ReadError or
    httpx.RemoteProtocolError ("Server disconnected without
    sending a response"). These are transient, so they are
    retried like a 503.
    """

    return isinstance(
        error,
        (httpx.TransportError, ConnectionError)
    )


def _usage(response) -> tuple[int | None, int | None, int | None]:
    """Read token counts from a Gemini response, if present."""

    usage = getattr(response, "usage_metadata", None)

    if usage is None:
        return None, None, None

    return (
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
        getattr(usage, "thoughts_token_count", None)
    )


async def generate_with_fallback(
    client,
    contents,
    config,
    models=None,
    max_attempts_per_model=2,
    base_delay=3,
    timeout_s=60
) -> LLMResult:
    """
    Generate content using multiple Gemini models.

    Policy:
      - 500/502/503/504, timeout, or a network error such as
        a dropped connection: retry the same model with
        exponential backoff, then move to the next model.
      - 429 (quota / rate limit): move to the next model
        immediately.
      - Any other error: raise immediately.

    The call is non-blocking (uses client.aio), so other
    asyncio tasks can run while waiting for Gemini.

    Returns an LLMResult. If every model fails, the last
    error is raised.
    """

    if models is None:
        models = DEFAULT_MODELS

    started = time.perf_counter()

    last_error = None
    attempts = 0
    failures = []

    for model in models:

        print(f"\nTrying model: {model}")

        for attempt in range(
            1,
            max_attempts_per_model + 1
        ):

            attempts += 1

            try:

                print(
                    f"Attempt "
                    f"{attempt}/{max_attempts_per_model}"
                )

                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    ),
                    timeout=timeout_s
                )

                print(
                    f"Success with {model}"
                )

                (
                    input_tokens,
                    output_tokens,
                    thinking_tokens
                ) = _usage(response)

                return LLMResult(
                    response=response,
                    model=model,
                    attempts=attempts,
                    latency_s=time.perf_counter() - started,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    thinking_tokens=thinking_tokens,
                    failures=failures
                )

            except Exception as e:

                last_error = e

                code = _error_code(e)

                print(
                    f"{model} attempt "
                    f"{attempt} failed:"
                )

                print(str(e))

                failures.append({
                    "model": model,
                    "attempt": attempt,
                    "code": code,
                    "type": type(e).__name__,
                    "error": str(e)[:300]
                })

                if code in QUOTA_CODES:

                    print(
                        f"{model} quota/rate limit hit. "
                        f"Skipping to next model."
                    )

                    break

                if (
                    code not in RETRYABLE_CODES
                    and not _is_network_error(e)
                ):
                    e.llm_failures = failures
                    raise

                if attempt < max_attempts_per_model:

                    delay = base_delay * (2 ** (attempt - 1))

                    print(
                        f"Retrying {model} "
                        f"in {delay} seconds..."
                    )

                    await asyncio.sleep(delay)

        print(
            f"{model} unavailable. "
            f"Trying next model..."
        )

    # Attach the failure history so callers can still record
    # what was tried, even though no result was produced.
    last_error.llm_failures = failures

    raise last_error