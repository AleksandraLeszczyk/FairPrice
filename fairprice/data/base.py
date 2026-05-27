import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import diskcache

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket limiter — sleeps when calls come in too fast."""

    def __init__(self, calls_per_minute: int):
        self._interval = 60.0 / calls_per_minute
        self._last: float = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < self._interval:
            time.sleep(self._interval - delta)
        self._last = time.monotonic()


class CachedClient:
    """
    Base for all data clients.

    Subclasses call ``_fetch_cached(key, fn, *args, **kwargs)`` and get
    transparent disk-based caching with configurable TTL.
    """

    def __init__(self, cache_dir: Path, default_ttl: int):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(cache_dir))
        self._default_ttl = default_ttl

    def _fetch_cached(
        self,
        key: str,
        fn: Callable,
        *args: Any,
        ttl: Optional[int] = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> Any:
        if not force_refresh and key in self._cache:
            logger.debug("cache hit: %s", key)
            return self._cache[key]

        logger.debug("cache miss: %s", key)
        result = fn(*args, **kwargs)
        if result is not None:
            self._cache.set(key, result, expire=ttl or self._default_ttl)
        return result

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("%s cache cleared", self.__class__.__name__)
