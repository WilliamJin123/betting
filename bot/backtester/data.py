"""
Resolved Market Data Fetcher
==============================

Fetches historically resolved markets from the Polymarket Gamma API for
backtesting.  Resolved markets have known outcomes, so we can test strategy
performance without waiting for live resolution.

Key limitation:
    The Gamma API returns *final* outcome prices (0 or 1) for closed markets,
    NOT the pre-resolution trading prices.  We cannot recover what the market
    price was at, say, 24 hours before resolution.  The backtesting engine
    handles this by using a proxy (default: 50/50 assumption).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResolvedMarket:
    """
    A single resolved (closed) Polymarket market with a known outcome.

    Fields:
        question:        The market's question text.
        condition_id:    Unique identifier for the market.
        outcomes:        Outcome labels, e.g. ["Yes", "No"].
        final_prices:    Final prices after resolution, e.g. [1.0, 0.0].
        winning_outcome: The label of the winning outcome, e.g. "Yes".
        end_date:        ISO 8601 end date string.
        volume:          Total USD volume traded.
        category:        Category tag if available (e.g. "sports", "crypto").
    """
    question: str
    condition_id: str
    outcomes: list[str] = field(default_factory=list)
    final_prices: list[float] = field(default_factory=list)
    winning_outcome: str = ""
    end_date: str = ""
    volume: float = 0.0
    category: str = ""


# ---------------------------------------------------------------------------
# Fetching logic
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    """Build an HTTP session with retries, matching the main client pattern."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "polymarket-bot/0.1",
    })
    return session


def fetch_resolved_markets(
    min_volume: float = 10_000,
    limit: int = 500,
    category: Optional[str] = None,
) -> list[ResolvedMarket]:
    """
    Fetch closed/resolved markets from the Polymarket Gamma API.

    The API is paginated (max 100 per page), so we loop until we have
    enough markets or run out of results.

    Args:
        min_volume: Minimum total volume in USD.  Higher volume markets have
                    more accurate prices and more meaningful outcomes.
        limit:      Maximum number of resolved markets to return.
        category:   Optional category filter (not officially supported by
                    the Gamma API, but we filter client-side if provided).

    Returns:
        List of ResolvedMarket objects, sorted by volume descending.
    """
    session = _build_session()
    all_markets: list[ResolvedMarket] = []
    page_size = 100
    max_pages = (limit // page_size) + 5  # fetch extra pages to account for filtering

    for page in range(max_pages):
        if len(all_markets) >= limit:
            break

        offset = page * page_size
        params: dict = {
            "limit": page_size,
            "offset": offset,
            "closed": True,
            "order": "volume",
            "ascending": False,
        }

        try:
            resp = session.get(f"{GAMMA_API_BASE}/markets", params=params, timeout=30)
            resp.raise_for_status()
            raw_markets = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Failed to fetch page %d: %s", page, e)
            break

        if not isinstance(raw_markets, list) or len(raw_markets) == 0:
            break

        for raw in raw_markets:
            if len(all_markets) >= limit:
                break

            market = _parse_resolved_market(raw)
            if market is None:
                continue

            # Volume filter
            if market.volume < min_volume:
                continue

            # Category filter (client-side)
            if category is not None and market.category.lower() != category.lower():
                continue

            all_markets.append(market)

        # If we got fewer results than page_size, no more pages
        if len(raw_markets) < page_size:
            break

    logger.info(
        "Fetched %d resolved markets (min_volume=%s, limit=%s)",
        len(all_markets), min_volume, limit,
    )
    return all_markets


def _parse_resolved_market(raw: dict) -> Optional[ResolvedMarket]:
    """
    Parse a raw Gamma API market dict into a ResolvedMarket.

    Returns None if:
      - The market isn't cleanly resolved (prices not 0/1)
      - Essential fields are missing
      - The market has no outcomes
    """
    # --- Parse outcomes ---
    outcomes: list[str] = []
    raw_outcomes = raw.get("outcomes")
    if isinstance(raw_outcomes, str):
        try:
            outcomes = json.loads(raw_outcomes)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw_outcomes, list):
        outcomes = raw_outcomes
    else:
        return None

    if not outcomes:
        return None

    # --- Parse outcome prices ---
    final_prices: list[float] = []
    raw_prices = raw.get("outcomePrices")
    if isinstance(raw_prices, str):
        try:
            final_prices = [float(p) for p in json.loads(raw_prices)]
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(raw_prices, list):
        try:
            final_prices = [float(p) for p in raw_prices]
        except (ValueError, TypeError):
            return None
    else:
        return None

    if len(final_prices) != len(outcomes):
        return None

    # --- Filter: only cleanly resolved markets (prices are 0 or 1) ---
    # Allow a small tolerance for floating-point weirdness
    tolerance = 0.02
    for price in final_prices:
        if not (price < tolerance or price > 1.0 - tolerance):
            return None

    # Exactly one outcome should have price ~1.0
    winners = [i for i, p in enumerate(final_prices) if p > 1.0 - tolerance]
    if len(winners) != 1:
        return None

    winning_idx = winners[0]
    winning_outcome = outcomes[winning_idx]

    # Snap prices to clean 0/1
    clean_prices = [1.0 if p > 1.0 - tolerance else 0.0 for p in final_prices]

    # --- Volume ---
    volume = float(raw.get("volumeNum", raw.get("volume", 0)) or 0)

    # --- Category ---
    # The Gamma API doesn't have a formal "category" field, but some markets
    # have tags or group slugs we can use.
    category = ""
    tags = raw.get("tags", [])
    if isinstance(tags, list) and tags:
        category = str(tags[0])
    elif raw.get("groupSlug"):
        category = str(raw.get("groupSlug", ""))

    # --- Build the dataclass ---
    return ResolvedMarket(
        question=raw.get("question", ""),
        condition_id=raw.get("conditionId", ""),
        outcomes=outcomes,
        final_prices=clean_prices,
        winning_outcome=winning_outcome,
        end_date=raw.get("endDate", raw.get("endDateIso", "")),
        volume=volume,
        category=category,
    )
