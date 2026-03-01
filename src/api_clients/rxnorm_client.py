"""RxNorm API wrapper for drug name resolution."""

import logging
from typing import Any

import httpx
from diskcache import Cache

from src.config import CACHE_DIR, RXNORM_BASE_URL

logger = logging.getLogger(__name__)

_cache = Cache(str(CACHE_DIR / "rxnorm"))


class RxNormClient:
    """Async client for the RxNorm REST API (drug name resolution only)."""

    async def _get_ingredient_rxcui(
        self, client: httpx.AsyncClient, rxcui: str
    ) -> tuple[str, str] | None:
        """Navigate from any RXCUI to its ingredient-level RXCUI and name.

        Args:
            client: Active httpx client.
            rxcui: Starting RXCUI (may be brand, SBD, etc.).

        Returns:
            Tuple of (ingredient_rxcui, generic_name) or None.
        """
        # Check properties to see if already an ingredient
        props_resp = await client.get(
            f"{RXNORM_BASE_URL}/rxcui/{rxcui}/properties.json",
        )
        props_resp.raise_for_status()
        properties = props_resp.json().get("properties", {})
        tty = properties.get("tty", "")

        if tty in ("IN", "MIN", "PIN"):
            return rxcui, properties.get("name", "").lower()

        # Not an ingredient — find the related ingredient
        related_resp = await client.get(
            f"{RXNORM_BASE_URL}/rxcui/{rxcui}/related.json",
            params={"tty": "IN"},
        )
        related_resp.raise_for_status()
        related_data = related_resp.json()

        for group in related_data.get("relatedGroup", {}).get("conceptGroup", []):
            for prop in group.get("conceptProperties", []):
                return prop["rxcui"], prop["name"].lower()

        # Fallback: return original with its name
        return rxcui, properties.get("name", "").lower()

    async def resolve_drug_name(self, name: str) -> dict[str, Any] | None:
        """Resolve a user-supplied drug name to its canonical info via RxNorm.

        Uses approximateTerm for fuzzy matching, then navigates to the
        ingredient-level concept for the canonical generic name and RXCUI.

        Args:
            name: Drug name as entered by the user (brand or generic).

        Returns:
            Dict with keys {rxcui, generic_name, brand_names} or None on failure.
        """
        cache_key = f"resolve:{name.lower()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Step 1: fuzzy match to get RXCUI
                resp = await client.get(
                    f"{RXNORM_BASE_URL}/approximateTerm.json",
                    params={"term": name, "maxEntries": 5},
                )
                resp.raise_for_status()
                data = resp.json()

                candidates = (
                    data.get("approximateGroup", {}).get("candidate", [])
                )
                if not candidates:
                    logger.error("resolve_drug_name(%s): no candidates", name)
                    return None

                rxcui = candidates[0].get("rxcui")
                if not rxcui:
                    logger.error("resolve_drug_name(%s): no rxcui in candidate", name)
                    return None

                # Step 2: navigate to ingredient-level RXCUI
                ingredient = await self._get_ingredient_rxcui(client, rxcui)
                if not ingredient:
                    logger.error("resolve_drug_name(%s): no ingredient found", name)
                    return None

                ing_rxcui, generic_name = ingredient

                result: dict[str, Any] = {
                    "rxcui": ing_rxcui,
                    "generic_name": generic_name,
                    "brand_names": [],
                }

                # Cache indefinitely for drug name resolution
                _cache.set(cache_key, result)
                return result

        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.error("resolve_drug_name(%s) failed: %s", name, exc)
            return None
