"""HTTP client for scraping RealGM.

Uses curl-cffi to impersonate Chrome and bypass Cloudflare Bot Management.
httpx is NOT suitable here — it fails Cloudflare's TLS fingerprint check.
"""
import concurrent.futures
import logging
import random
import time

from curl_cffi import requests as cf_requests
from curl_cffi.requests.exceptions import Timeout, DNSError, ConnectionError as CurlConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Chrome version to impersonate — update when curl-cffi adds newer versions
IMPERSONATE = "chrome120"


class RateLimitError(Exception):
    """Raised when the server returns 429 or 503."""
    pass


class RealGMClient:
    """
    HTTP client for RealGM with Cloudflare bypass, rate limiting, and retry.

    Uses curl-cffi's Chrome impersonation to pass Cloudflare Bot Management.

    Args:
        headers: Additional request headers (merged with curl-cffi defaults).
        rate_limit_rps: Requests per second limit.
        jitter: Maximum random jitter in seconds added to each sleep.
    """

    # Hard wall-clock deadline per request, enforced via ThreadPoolExecutor.
    # curl-cffi's own timeout param is unreliable for stalled connections.
    REQUEST_TIMEOUT_S = 45

    def __init__(self, headers: dict, rate_limit_rps: float = 1.0, jitter: float = 0.3):
        self.headers = headers
        self.rate_limit_rps = rate_limit_rps
        self.jitter = jitter
        self._last_request_time: float = 0.0
        self._session = cf_requests.Session(impersonate=IMPERSONATE)
        self._session.headers.update(headers)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _rate_limit_sleep(self) -> None:
        min_interval = 1.0 / self.rate_limit_rps
        elapsed = time.monotonic() - self._last_request_time
        sleep_time = min_interval + random.uniform(0, self.jitter) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    def get(self, url: str) -> str:
        """Fetch URL, enforce rate limit, retry on transient errors."""
        return self._get_with_retry(url)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=2, max=120),
        retry=retry_if_exception_type((Timeout, DNSError, CurlConnectionError, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get_with_retry(self, url: str) -> str:
        self._rate_limit_sleep()
        logger.debug("GET %s", url)
        self._last_request_time = time.monotonic()
        future = self._executor.submit(self._session.get, url)
        try:
            response = future.result(timeout=self.REQUEST_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise Timeout(f"Hard timeout ({self.REQUEST_TIMEOUT_S}s) exceeded for {url}")
        if response.status_code in (429, 503):
            logger.warning("Rate limit hit (HTTP %d): %s", response.status_code, url)
            raise RateLimitError(f"HTTP {response.status_code}")
        if response.status_code == 403:
            # 403 from Cloudflare means impersonation failed or IP is blocked
            raise RateLimitError(f"HTTP 403 (Cloudflare) for {url}")
        response.raise_for_status()
        return response.text

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


