"""OpenFDA Drug Label API wrapper."""

import logging
from typing import Any

import httpx
from diskcache import Cache

from src.config import CACHE_DIR, OPENFDA_API_KEY, OPENFDA_BASE_URL

logger = logging.getLogger(__name__)

_cache = Cache(str(CACHE_DIR / "openfda"))
_LABEL_TTL = 86400  # 24 hours


class OpenFDAClient:
    """Async client for the openFDA Drug Label API."""

    async def get_top_drugs(self, limit: int = 500) -> list[str]:
        """Fetch the most frequently reported generic drug names.

        Args:
            limit: Maximum number of drug names to return.

        Returns:
            List of lowercase generic drug name strings.
        """
        cache_key = f"top_drugs:{limit}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        params: dict[str, Any] = {
            "count": "openfda.generic_name.exact",
            "limit": limit,
        }
        if OPENFDA_API_KEY:
            params["api_key"] = OPENFDA_API_KEY

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(OPENFDA_BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            names = [
                r["term"].lower() for r in data.get("results", []) if "term" in r
            ]
            _cache.set(cache_key, names, expire=_LABEL_TTL)
            return names
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.error("get_top_drugs failed: %s", exc)
            return []

    async def search_labels(self, drug_name: str, limit: int = 2) -> list[dict]:
        """Search for drug labels by generic name.

        Args:
            drug_name: Generic drug name to search for.
            limit: Maximum number of label results.

        Returns:
            List of raw label dicts from the openFDA response.
        """
        cache_key = f"labels:{drug_name.lower()}:{limit}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        params: dict[str, Any] = {
            "search": f'openfda.generic_name:"{drug_name}"',
            "limit": limit,
        }
        if OPENFDA_API_KEY:
            params["api_key"] = OPENFDA_API_KEY

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(OPENFDA_BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            _cache.set(cache_key, results, expire=_LABEL_TTL)
            return results
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.error("search_labels(%s) failed: %s", drug_name, exc)
            return []
