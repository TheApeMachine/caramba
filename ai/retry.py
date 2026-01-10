"""Retry utilities with exponential backoff.

Provides robust retry logic for HTTP requests and other async operations.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

T = TypeVar("T")

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 0.5  # seconds
DEFAULT_MAX_DELAY = 10.0  # seconds
DEFAULT_JITTER = 0.1  # 10% jitter


def calculate_backoff(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: The current attempt number (0-indexed).
        base_delay: The base delay in seconds.
        max_delay: Maximum delay cap in seconds.
        jitter: Jitter factor (0.0 to 1.0).

    Returns:
        Delay in seconds to wait before next attempt.
    """
    # Exponential backoff: base * 2^attempt
    delay = base_delay * (2 ** attempt)

    # Cap at max delay
    delay = min(delay, max_delay)

    # Add random jitter
    jitter_amount = delay * jitter * random.random()
    delay += jitter_amount

    return delay


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Tuple of exception types to retry on.

    Returns:
        Result of the function if successful.

    Raises:
        The last exception if all retries fail.
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e

            if attempt < max_retries:
                delay = calculate_backoff(attempt, base_delay, max_delay)
                await asyncio.sleep(delay)

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed with no exception")


async def http_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 5.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> tuple[bool, int, str]:
    """Perform HTTP GET with retry and exponential backoff.

    Args:
        client: The httpx async client.
        url: The URL to request.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        base_delay: Base delay between retries.

    Returns:
        Tuple of (success, status_code, error_message).
        If successful, status_code is the HTTP status and error_message is empty.
        If failed, status_code is 0 and error_message contains the error.
    """
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            response = await client.get(url, timeout=timeout)
            return True, response.status_code, ""
        except httpx.ConnectError:
            last_error = "connection refused"
        except httpx.TimeoutException:
            last_error = "timeout"
        except httpx.HTTPStatusError as e:
            # Don't retry on HTTP errors (4xx, 5xx)
            return False, e.response.status_code, f"HTTP {e.response.status_code}"
        except Exception as e:
            error_str = str(e)
            if len(error_str) > 30:
                error_str = error_str[:27] + "..."
            last_error = error_str

        # Wait before retry (except on last attempt)
        if attempt < max_retries:
            delay = calculate_backoff(attempt, base_delay)
            await asyncio.sleep(delay)

    return False, 0, last_error


async def http_get_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 10.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> tuple[bool, dict[str, Any] | None, str]:
    """Perform HTTP GET and parse JSON with retry.

    Args:
        client: The httpx async client.
        url: The URL to request.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        base_delay: Base delay between retries.

    Returns:
        Tuple of (success, json_data, error_message).
    """
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            response = await client.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, response.json(), ""
            last_error = f"HTTP {response.status_code}"
        except httpx.ConnectError:
            last_error = "connection refused"
        except httpx.TimeoutException:
            last_error = "timeout"
        except Exception as e:
            error_str = str(e)
            if len(error_str) > 30:
                error_str = error_str[:27] + "..."
            last_error = error_str

        # Wait before retry (except on last attempt)
        if attempt < max_retries:
            delay = calculate_backoff(attempt, base_delay)
            await asyncio.sleep(delay)

    return False, None, last_error
