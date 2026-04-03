import hashlib
import json
import logging
from pathlib import Path

import diskcache

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(".cache") / "serpapi"
_TTL_SECONDS = 6 * 60 * 60  # 6 hours

_cache = diskcache.Cache(str(_CACHE_DIR))


def _make_key(params: dict) -> str:
    """Deterministic cache key from API params, excluding the secret key."""
    filtered = {k: v for k, v in sorted(params.items()) if k != "api_key"}
    raw = json.dumps(filtered, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(params: dict) -> dict | None:
    key = _make_key(params)
    result = _cache.get(key)
    if result is not None:
        logger.info("Cache HIT for %s (engine=%s)", key[:12], params.get("engine"))
    else:
        logger.info("Cache MISS for %s (engine=%s)", key[:12], params.get("engine"))
    return result


def set_cached(params: dict, result: dict) -> None:
    key = _make_key(params)
    _cache.set(key, result, expire=_TTL_SECONDS)
    logger.info("Cache SET for %s (ttl=%ds)", key[:12], _TTL_SECONDS)
