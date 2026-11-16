"""HTTP client with rate limiting and retries, shared by every fetcher."""

import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from tsfm.constants import (
    HTTP_BACKOFF_MAX_SECONDS,
    HTTP_BACKOFF_SECONDS,
    HTTP_ERROR_BODY_CHARS,
    HTTP_MAX_ATTEMPTS,
    HTTP_RETRY_STATUS_CODES,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
)
from tsfm.log import get_logger

logger = get_logger(__name__)

type Params = Mapping[str, str | int | float]
# Fixed headers, or a function called once per attempt for headers that must differ between retries.
type Headers = Mapping[str, str] | Callable[[], Mapping[str, str]]


class RetryableStatusError(Exception):
    """The server answered with a status worth retrying."""


class RateLimiter:
    """
    Space calls at least ``1 / per_second`` seconds apart.

    Parameters
    ----------
    per_second : float
        Maximum call rate.
    sleep : Callable[[float], None], default: time.sleep
        Sleep function, replaceable in tests.
    """

    def __init__(self, per_second: float, sleep: Callable[[float], None] = time.sleep) -> None:
        self._interval = 1.0 / per_second
        self._sleep = sleep
        self._next_call = 0.0

    def wait(self) -> None:
        """Block until the next call is allowed."""
        now = time.monotonic()
        if now < self._next_call:
            self._sleep(self._next_call - now)
        self._next_call = max(now, self._next_call) + self._interval


class HttpClient:
    """
    GET client that rate-limits, times out and retries transient failures.

    Parameters
    ----------
    requests_per_second : float
        Rate limit applied before every request.
    headers : Mapping[str, str] or None, default: None
        Headers sent with every request, e.g. authorisation.
    transport : httpx.BaseTransport or None, default: None
        Transport override; tests pass ``httpx.MockTransport``.
    sleep : Callable[[float], None], default: time.sleep
        Sleep used by the rate limiter and the retry backoff.

    Examples
    --------
    >>> with HttpClient(requests_per_second=2.0) as client:  # doctest: +SKIP
    ...     payload = client.get_json("https://api.example.com/items", {"limit": 10})
    """

    def __init__(
        self,
        requests_per_second: float,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": HTTP_USER_AGENT, **(headers or {})},
            timeout=HTTP_TIMEOUT_SECONDS,
            transport=transport,
            follow_redirects=True,
        )
        self._limiter = RateLimiter(requests_per_second, sleep)
        self._retrying = Retrying(
            retry=retry_if_exception_type((httpx.TransportError, RetryableStatusError)),
            stop=stop_after_attempt(HTTP_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=HTTP_BACKOFF_SECONDS, max=HTTP_BACKOFF_MAX_SECONDS),
            sleep=sleep,
            reraise=True,
        )

    def __enter__(self) -> Self:
        """
        Enter the context.

        Returns
        -------
        Self
            This client.
        """
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """
        Close the client on leaving the context.

        Parameters
        ----------
        exc_type : type[BaseException] or None
            Exception type, if one was raised.
        exc : BaseException or None
            Exception, if one was raised.
        tb : TracebackType or None
            Traceback, if one was raised.
        """
        self.close()

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()

    def get(self, url: str, params: Params | None = None, headers: Headers | None = None) -> httpx.Response:
        """
        Send a GET request, retrying transient failures.

        Parameters
        ----------
        url : str
            Absolute URL.
        params : Params or None, default: None
            Query parameters.
        headers : Headers or None, default: None
            Extra headers for this request only.

        Returns
        -------
        httpx.Response
            A successful response.

        Raises
        ------
        httpx.HTTPStatusError
            On a non-retryable error status.
        RetryableStatusError
            If a retryable status persists after the last attempt.
        """
        return self._retrying(self._get_once, url, params, headers)

    def get_json(self, url: str, params: Params | None = None, headers: Headers | None = None) -> Any:
        """
        Send a GET request and decode the JSON body.

        Parameters
        ----------
        url : str
            Absolute URL.
        params : Params or None, default: None
            Query parameters.
        headers : Headers or None, default: None
            Extra headers for this request only.

        Returns
        -------
        Any
            Decoded JSON.
        """
        return self.get(url, params, headers).json()

    def _get_once(self, url: str, params: Params | None, headers: Headers | None) -> httpx.Response:
        """
        Send one rate-limited GET request.

        Parameters
        ----------
        url : str
            Absolute URL.
        params : Params or None
            Query parameters.
        headers : Headers or None
            Extra headers, built anew for this attempt if given as a function.

        Returns
        -------
        httpx.Response
            A successful response.

        Raises
        ------
        RetryableStatusError
            On a status in ``HTTP_RETRY_STATUS_CODES``.
        httpx.HTTPStatusError
            On any other error status.
        """
        self._limiter.wait()
        response = self._client.get(url, params=params, headers=headers() if callable(headers) else headers)
        if response.status_code in HTTP_RETRY_STATUS_CODES:
            logger.warning("http_retryable_status", url=url, status=response.status_code)
            raise RetryableStatusError(f"{response.status_code} from {url}")
        if response.is_error:
            # httpx's own message prints the query string, where FRED and EIA carry the API key.
            message = f"{response.status_code} from {url}: {response.text[:HTTP_ERROR_BODY_CHARS]}"
            raise httpx.HTTPStatusError(message, request=response.request, response=response)
        return response
