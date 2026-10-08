import asyncio


async def generate_with_fallback(
    client,
    contents,
    config,
    models=None,
    max_attempts_per_model=2,
    base_delay=3
):
    """
    Generate content using multiple Gemini models.

    If a model temporarily fails, the next model
    is attempted.
    """

    if models is None:
        models = [
            
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite"
        ]

    last_error = None

    for model in models:

        print(f"\nTrying model: {model}")

        for attempt in range(
            1,
            max_attempts_per_model + 1
        ):

            try:

                print(
                    f"Attempt "
                    f"{attempt}/{max_attempts_per_model}"
                )

                response = (
                    client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )
                )

                print(
                    f"Success with {model}"
                )

                return response

            except Exception as e:

                last_error = e

                error_text = str(e)

                print(
                    f"{model} attempt "
                    f"{attempt} failed:"
                )

                print(error_text)

                # Only retry temporary errors.
                is_temporary = (
                    "503" in error_text
                    or "429" in error_text
                    or "500" in error_text
                    or "502" in error_text
                    or "504" in error_text
                )

                if not is_temporary:
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

    raise last_error