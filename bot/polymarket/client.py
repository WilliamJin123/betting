"""
Polymarket API Client (Read-Only)
=================================

Wraps the two Polymarket APIs for reading market data:

    1. Gamma API (https://gamma-api.polymarket.com)
       - Market discovery: search/filter/list all markets
       - Rich metadata: question text, descriptions, volume, liquidity, dates
       - Good for: "show me all active markets with > $10k volume"

    2. CLOB API (https://clob.polymarket.com)
       - Live trading data: order books, best prices, spreads
       - Market structure: tick sizes, token IDs, fee rates
       - Good for: "what's the current bid/ask for this specific market?"

Why two APIs?
    Gamma is Polymarket's metadata/discovery layer. CLOB is the trading engine.
    To build a bot, you need both: Gamma to find interesting markets, CLOB to
    see the live prices and eventually place trades.

Authentication:
    All endpoints used here are public (no API key needed). You only need
    auth for placing orders, which is Phase 2.

Rate Limiting:
    We add a small delay between requests and retry on 429 (Too Many Requests).
    Polymarket doesn't publish official rate limits, but the community consensus
    is ~10 requests/second is safe.

Usage:
    from bot.polymarket.client import PolymarketClient

    client = PolymarketClient()

    # Find active markets with good liquidity
    markets = client.get_active_markets(min_volume=1000, min_liquidity=500)

    # Get order book for a specific token
    book = client.get_order_book(markets[0].yes_token_id)

    # Get current price
    price = client.get_price(markets[0].yes_token_id, side="BUY")
"""

import json
import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bot.polymarket.models import Market, OrderBook, OrderBookLevel, Token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Pagination end marker for the CLOB API. When next_cursor equals this,
# there are no more pages.
END_CURSOR = "LTE="

# Default first-page cursor (base64-encoded "0").
FIRST_CURSOR = "MA=="

# Be polite: minimum seconds between consecutive requests.
REQUEST_DELAY = 0.1


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PolymarketClient:
    """
    Read-only client for Polymarket's public APIs.

    This is the main class you interact with. It handles:
        - HTTP session management (connection pooling, retries)
        - Rate limiting (so we don't get banned)
        - Parsing raw JSON into our dataclass models
        - Pagination (fetching all pages of results)

    Args:
        request_delay: Seconds to wait between requests (default 0.1).
                       Increase if you're getting 429 errors.
    """

    def __init__(self, request_delay: float = REQUEST_DELAY):
        self.request_delay = request_delay
        self.gamma_base = GAMMA_API_BASE
        self.clob_base = CLOB_API_BASE
        self._last_request_time: float = 0.0
        self.session = self._build_session()

    # ----- Session Setup -----

    @staticmethod
    def _build_session() -> requests.Session:
        """
        Build an HTTP session with automatic retries.

        Why retries?
            Network blips happen. A 500 error from the server is usually
            temporary. Instead of crashing, we wait a bit and try again.

        Retry strategy:
            - Up to 3 retries
            - Exponential backoff (1s, 2s, 4s)
            - Only retry on 429 (rate limit), 500, 502, 503, 504 (server errors)
            - Don't retry on 4xx client errors (those mean we did something wrong)
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "Accept": "application/json",
            "User-Agent": "polymarket-bot/0.1",
        })
        return session

    # ----- Rate Limiting -----

    def _throttle(self) -> None:
        """
        Enforce minimum delay between requests.

        Simple but effective: track when we last made a request, sleep if
        we're going too fast. This keeps us well under rate limits.
        """
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.monotonic()

    # ----- Low-level HTTP -----

    def _get(self, url: str, params: Optional[dict] = None) -> dict | list:
        """
        Make a GET request with throttling and error handling.

        Returns the parsed JSON response (dict or list).
        Raises requests.HTTPError on 4xx/5xx after retries are exhausted.
        """
        self._throttle()
        logger.debug(f"GET {url} params={params}")
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, url: str, json_body: Optional[dict | list] = None) -> dict | list:
        """
        Make a POST request with throttling and error handling.

        Some CLOB endpoints use POST for batch queries (e.g., getting
        prices for multiple tokens at once).
        """
        self._throttle()
        logger.debug(f"POST {url} body={json_body}")
        resp = self.session.post(url, json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ======================================================================
    # GAMMA API — Market Discovery & Metadata
    # ======================================================================

    def get_gamma_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: Optional[bool] = True,
        closed: Optional[bool] = False,
        order: str = "volume24hr",
        ascending: bool = False,
    ) -> list[Market]:
        """
        Fetch markets from the Gamma API with filtering and sorting.

        This is the best way to discover markets. The Gamma API has rich
        filtering — you can search by activity status, sort by volume, etc.

        Args:
            limit:     Max markets per page (up to 100).
            offset:    Skip this many results (for pagination).
            active:    Only return active markets (True) or include all (None).
            closed:    Only return closed markets (True), open (False), or all (None).
            order:     Sort field. Common values:
                       - "volume24hr" (most traded today)
                       - "liquidity" (most liquid)
                       - "volume" (most traded all-time)
                       - "endDate" (closing soonest)
                       - "startDate" (newest)
            ascending: Sort direction. False = descending (highest first).

        Returns:
            List of Market objects with metadata populated from Gamma.
        """
        params: dict = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": ascending,
        }
        if active is not None:
            params["active"] = active
        if closed is not None:
            params["closed"] = closed

        raw_markets = self._get(f"{GAMMA_API_BASE}/markets", params=params)

        # Gamma returns a plain JSON array (not wrapped in {"data": ...}).
        if not isinstance(raw_markets, list):
            logger.warning(f"Unexpected Gamma response type: {type(raw_markets)}")
            return []

        return [self._parse_gamma_market(m) for m in raw_markets]

    def get_active_markets(
        self,
        min_volume: float = 0,
        min_liquidity: float = 0,
        limit: int = 100,
        sort_by: str = "volume24hr",
    ) -> list[Market]:
        """
        Get active, open markets filtered by volume and liquidity.

        This is the convenience method you'll use most often. It fetches
        active markets sorted by recent trading activity, then filters out
        anything below your volume/liquidity thresholds.

        Why filter?
            Low-volume markets have unreliable prices. Low-liquidity markets
            have high slippage (your trade moves the price against you).
            For a $10 bankroll, we want markets where we can get in/out easily.

        Args:
            min_volume:    Minimum total volume in USD (0 = no filter).
            min_liquidity: Minimum current liquidity in USD (0 = no filter).
            limit:         Max results to fetch from API.
            sort_by:       Sort field (default: 24h volume).

        Returns:
            List of Market objects, filtered and sorted.
        """
        markets = self.get_gamma_markets(
            limit=limit,
            active=True,
            closed=False,
            order=sort_by,
            ascending=False,
        )
        return [
            m for m in markets
            if m.volume >= min_volume and m.liquidity >= min_liquidity
        ]

    def get_all_active_markets(
        self,
        min_volume: float = 0,
        min_liquidity: float = 0,
        max_pages: int = 10,
    ) -> list[Market]:
        """
        Paginate through ALL active markets from the Gamma API.

        The Gamma API returns max 100 markets per request. This method
        fetches multiple pages until we run out of results or hit max_pages.

        Args:
            min_volume:    Minimum total volume in USD.
            min_liquidity: Minimum current liquidity in USD.
            max_pages:     Safety limit on number of pages to fetch.

        Returns:
            All active markets matching the filters.
        """
        all_markets: list[Market] = []
        page_size = 100

        for page in range(max_pages):
            offset = page * page_size
            batch = self.get_gamma_markets(
                limit=page_size,
                offset=offset,
                active=True,
                closed=False,
                order="volume24hr",
                ascending=False,
            )
            if not batch:
                break

            for m in batch:
                if m.volume >= min_volume and m.liquidity >= min_liquidity:
                    all_markets.append(m)

            # If we got fewer than page_size, we've reached the end.
            if len(batch) < page_size:
                break

        logger.info(f"Fetched {len(all_markets)} active markets across {page + 1} pages")
        return all_markets

    def search_markets(self, query: str, limit: int = 20) -> list[Market]:
        """
        Search markets by keyword using the Gamma API.

        Useful for finding specific markets like "Bitcoin" or "Trump".

        Note: This uses the Gamma API's slug-based filtering. It's not
        a full-text search — it matches against the market slug (URL-friendly
        version of the question).

        Args:
            query:  Search term (e.g., "bitcoin", "election", "fed").
            limit:  Max results.

        Returns:
            List of matching Market objects.
        """
        params = {
            "limit": limit,
            "active": True,
            "closed": False,
            "slug": query.lower().replace(" ", "-"),
        }
        raw_markets = self._get(f"{GAMMA_API_BASE}/markets", params=params)
        if not isinstance(raw_markets, list):
            return []
        return [self._parse_gamma_market(m) for m in raw_markets]

    def get_gamma_market_by_id(self, market_id: str) -> Optional[Market]:
        """
        Fetch a single market by its Gamma ID (numeric string).

        Args:
            market_id: The Gamma market ID (e.g., "531202").

        Returns:
            Market object, or None if not found.
        """
        try:
            raw = self._get(f"{GAMMA_API_BASE}/markets/{market_id}")
            if isinstance(raw, dict):
                return self._parse_gamma_market(raw)
            return None
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    def get_gamma_market_by_slug(self, slug: str) -> Optional[Market]:
        """
        Fetch a single market by its slug (the URL-friendly name).

        Args:
            slug: Market slug (e.g., "will-bitcoin-hit-100k").

        Returns:
            Market object, or None if not found.
        """
        params = {"slug": slug, "limit": 1}
        raw = self._get(f"{GAMMA_API_BASE}/markets", params=params)
        if isinstance(raw, list) and raw:
            return self._parse_gamma_market(raw[0])
        return None

    # ======================================================================
    # CLOB API — Live Prices & Order Books
    # ======================================================================

    def get_clob_markets(self, next_cursor: str = FIRST_CURSOR) -> tuple[list[Market], str]:
        """
        Fetch one page of markets from the CLOB API.

        The CLOB API has less metadata than Gamma but includes live trading
        state (whether order book is enabled, token prices, etc.).

        Pagination:
            The CLOB uses cursor-based pagination. Pass the returned cursor
            to the next call. When the cursor equals "LTE=", you've reached
            the end.

        Args:
            next_cursor: Pagination cursor (default: start from beginning).

        Returns:
            Tuple of (list of Markets, next_cursor string).
            If next_cursor == "LTE=", there are no more pages.
        """
        raw = self._get(
            f"{CLOB_API_BASE}/markets",
            params={"next_cursor": next_cursor},
        )

        # The CLOB wraps results in {"data": [...], "next_cursor": "..."}
        if isinstance(raw, dict):
            data = raw.get("data", [])
            cursor = raw.get("next_cursor", END_CURSOR)
        else:
            data = raw if isinstance(raw, list) else []
            cursor = END_CURSOR

        markets = [self._parse_clob_market(m) for m in data]
        return markets, cursor

    def get_clob_market(self, condition_id: str) -> Optional[Market]:
        """
        Fetch a single market from the CLOB API by condition_id.

        Args:
            condition_id: The market's condition ID (hex string starting with 0x).

        Returns:
            Market object, or None if not found.
        """
        try:
            raw = self._get(f"{CLOB_API_BASE}/markets/{condition_id}")
            if isinstance(raw, dict):
                return self._parse_clob_market(raw)
            return None
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    # ----- Order Book -----

    def get_order_book(self, token_id: str) -> OrderBook:
        """
        Get the full order book snapshot for a specific outcome token.

        This shows all open buy orders (bids) and sell orders (asks).
        It's the most detailed view of market liquidity.

        Args:
            token_id: The outcome token's ID (long numeric string).
                      Get this from market.yes_token_id or market.tokens[i].token_id.

        Returns:
            OrderBook with bids, asks, spread, and midpoint calculated.
        """
        raw = self._get(f"{CLOB_API_BASE}/book", params={"token_id": token_id})
        return self._parse_order_book(raw)

    # ----- Price Endpoints -----

    def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """
        Get the current best price for a token.

        "Best price" depends on the side:
            - BUY:  The lowest price someone is willing to sell at (best ask).
                    This is what you'd pay to buy shares right now.
            - SELL: The highest price someone is willing to buy at (best bid).
                    This is what you'd receive for selling shares right now.

        Args:
            token_id: The outcome token's ID.
            side:     "BUY" or "SELL".

        Returns:
            Price as a float (0.00 to 1.00), or None if no orders exist.
        """
        raw = self._get(
            f"{CLOB_API_BASE}/price",
            params={"token_id": token_id, "side": side},
        )
        price_str = raw.get("price") if isinstance(raw, dict) else None
        if price_str is not None:
            return float(price_str)
        return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """
        Get the midpoint price for a token.

        Midpoint = (best_bid + best_ask) / 2

        This is often considered the "fair" price because it's in the middle
        of what buyers and sellers are willing to trade at. It's what most
        pricing models use.

        Args:
            token_id: The outcome token's ID.

        Returns:
            Midpoint price as a float, or None if unavailable.
        """
        raw = self._get(
            f"{CLOB_API_BASE}/midpoint",
            params={"token_id": token_id},
        )
        mid_str = raw.get("mid") if isinstance(raw, dict) else None
        if mid_str is not None:
            return float(mid_str)
        return None

    def get_spread(self, token_id: str) -> Optional[float]:
        """
        Get the bid-ask spread for a token.

        Spread = best_ask - best_bid

        A tight spread (e.g., 0.01) means the market is liquid — buyers and
        sellers nearly agree on the price. A wide spread (e.g., 0.10) means
        the market is thin and you'll pay a premium to trade.

        For our $10 bot, we should avoid markets with spreads > 0.05 (5 cents).

        Args:
            token_id: The outcome token's ID.

        Returns:
            Spread as a float, or None if unavailable.
        """
        raw = self._get(
            f"{CLOB_API_BASE}/spread",
            params={"token_id": token_id},
        )
        spread_str = raw.get("spread") if isinstance(raw, dict) else None
        if spread_str is not None:
            return float(spread_str)
        return None

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """
        Get the price of the most recent trade for a token.

        This tells you what someone actually paid last, not what the current
        bid/ask is. Useful for detecting stale markets (if last trade was
        hours ago at a very different price, something may be off).

        Args:
            token_id: The outcome token's ID.

        Returns:
            Last trade price as a float, or None if no trades.
        """
        raw = self._get(
            f"{CLOB_API_BASE}/last-trade-price",
            params={"token_id": token_id},
        )
        price_str = raw.get("price") if isinstance(raw, dict) else None
        if price_str is not None:
            return float(price_str)
        return None

    # ----- Batch Price Endpoints -----

    def get_prices_batch(self, token_ids: list[str], side: str = "BUY") -> dict[str, float]:
        """
        Get prices for multiple tokens in a single request.

        More efficient than calling get_price() in a loop. The CLOB API
        supports batch queries via POST.

        Args:
            token_ids: List of token IDs.
            side:      "BUY" or "SELL".

        Returns:
            Dict mapping token_id -> price.
        """
        if not token_ids:
            return {}

        # The /prices endpoint accepts a list of {token_id, side} objects.
        body = [{"token_id": tid, "side": side} for tid in token_ids]
        raw = self._post(f"{CLOB_API_BASE}/prices", json_body=body)

        result = {}
        if isinstance(raw, dict):
            # Response is {token_id: {side: price_string}, ...}
            # e.g. {"abc123": {"BUY": "0.65"}, "def456": {"BUY": "0.35"}}
            for tid, val in raw.items():
                try:
                    if isinstance(val, dict):
                        # Nested: {side: price_string}
                        price_str = val.get(side, val.get(side.upper()))
                        if price_str is not None:
                            result[tid] = float(price_str)
                    else:
                        # Flat: price_string directly
                        result[tid] = float(val)
                except (ValueError, TypeError):
                    pass
        return result

    def get_midpoints_batch(self, token_ids: list[str]) -> dict[str, float]:
        """
        Get midpoint prices for multiple tokens in a single request.

        Args:
            token_ids: List of token IDs.

        Returns:
            Dict mapping token_id -> midpoint price.
        """
        if not token_ids:
            return {}

        body = [{"token_id": tid} for tid in token_ids]
        raw = self._post(f"{CLOB_API_BASE}/midpoints", json_body=body)

        result = {}
        if isinstance(raw, dict):
            # Response is {token_id: {mid: price_string}, ...}
            for tid, val in raw.items():
                try:
                    if isinstance(val, dict):
                        mid_str = val.get("mid")
                        if mid_str is not None:
                            result[tid] = float(mid_str)
                    else:
                        result[tid] = float(val)
                except (ValueError, TypeError):
                    pass
        return result

    # ----- Utility Endpoints -----

    def get_server_time(self) -> Optional[str]:
        """
        Get the CLOB server's current timestamp.

        Useful for debugging clock skew issues.
        """
        raw = self._get(f"{CLOB_API_BASE}/time")
        return str(raw) if raw else None

    def health_check(self) -> bool:
        """
        Check if the CLOB API is up and responding.

        Returns True if the server responds, False otherwise.
        """
        try:
            self._get(f"{CLOB_API_BASE}/ok")
            return True
        except Exception:
            return False

    # ======================================================================
    # Parsing Helpers
    # ======================================================================

    @staticmethod
    def _parse_gamma_market(raw: dict) -> Market:
        """
        Convert a raw Gamma API market dict into our Market dataclass.

        The Gamma API returns camelCase fields with a LOT of data. We pick
        out just the fields we care about.
        """
        # Parse outcome prices from the string-encoded list.
        outcome_prices = []
        raw_prices = raw.get("outcomePrices")
        if isinstance(raw_prices, str):
            try:
                outcome_prices = [float(p) for p in json.loads(raw_prices)]
            except (json.JSONDecodeError, ValueError):
                pass
        elif isinstance(raw_prices, list):
            outcome_prices = [float(p) for p in raw_prices]

        # Parse outcomes from the string-encoded list.
        outcomes = []
        raw_outcomes = raw.get("outcomes")
        if isinstance(raw_outcomes, str):
            try:
                outcomes = json.loads(raw_outcomes)
            except json.JSONDecodeError:
                pass
        elif isinstance(raw_outcomes, list):
            outcomes = raw_outcomes

        # Build Token objects from clobTokenIds.
        tokens = []
        clob_ids = raw.get("clobTokenIds", [])
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except json.JSONDecodeError:
                clob_ids = []

        for i, tid in enumerate(clob_ids):
            tokens.append(Token(
                token_id=str(tid),
                outcome=outcomes[i] if i < len(outcomes) else f"Outcome {i}",
                price=outcome_prices[i] if i < len(outcome_prices) else 0.0,
            ))

        return Market(
            condition_id=raw.get("conditionId", ""),
            question=raw.get("question", ""),
            slug=raw.get("slug", ""),
            description=raw.get("description", ""),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            tokens=tokens,
            end_date=raw.get("endDate", raw.get("endDateIso", "")),
            active=bool(raw.get("active", False)),
            closed=bool(raw.get("closed", False)),
            volume=float(raw.get("volumeNum", raw.get("volume", 0))),
            liquidity=float(raw.get("liquidityNum", raw.get("liquidity", 0))),
            volume_24h=float(raw.get("volume24hr", 0)),
            spread=float(raw.get("spread", 0)),
            last_trade_price=float(raw.get("lastTradePrice", 0)),
            best_bid=float(raw.get("bestBid", 0)),
            best_ask=float(raw.get("bestAsk", 0)),
            min_order_size=float(raw.get("orderMinSize", 0)),
            tick_size=float(raw.get("orderPriceMinTickSize", 0.01)),
            neg_risk=bool(raw.get("negRisk", False)),
        )

    @staticmethod
    def _parse_clob_market(raw: dict) -> Market:
        """
        Convert a raw CLOB API market dict into our Market dataclass.

        The CLOB API uses snake_case and has different field names than Gamma.
        It has less metadata but more trading-specific fields.
        """
        tokens = []
        for t in raw.get("tokens", []):
            tokens.append(Token(
                token_id=str(t.get("token_id", "")),
                outcome=t.get("outcome", ""),
                price=float(t.get("price", 0)),
                winner=bool(t.get("winner", False)),
            ))

        outcome_prices = [t.price for t in tokens]
        outcomes = [t.outcome for t in tokens]

        return Market(
            condition_id=raw.get("condition_id", ""),
            question=raw.get("question", ""),
            slug=raw.get("market_slug", ""),
            description=raw.get("description", ""),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            tokens=tokens,
            end_date=raw.get("end_date_iso", ""),
            active=bool(raw.get("active", False)),
            closed=bool(raw.get("closed", False)),
            neg_risk=bool(raw.get("neg_risk", False)),
            min_order_size=float(raw.get("minimum_order_size", 0)),
            tick_size=float(raw.get("minimum_tick_size", 0.01)),
        )

    @staticmethod
    def _parse_order_book(raw: dict) -> OrderBook:
        """
        Convert a raw CLOB order book response into our OrderBook dataclass.

        The CLOB returns bids/asks as lists of {"price": "0.65", "size": "150"}.
        Note: prices and sizes come as strings, not numbers.
        """
        bids = []
        for entry in raw.get("bids", []):
            bids.append(OrderBookLevel(
                price=float(entry.get("price", 0)),
                size=float(entry.get("size", 0)),
            ))

        asks = []
        for entry in raw.get("asks", []):
            asks.append(OrderBookLevel(
                price=float(entry.get("price", 0)),
                size=float(entry.get("size", 0)),
            ))

        # Sort: bids highest first, asks lowest first.
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        best_bid = bids[0].price if bids else 0.0
        best_ask = asks[0].price if asks else 0.0

        spread = (best_ask - best_bid) if (best_bid > 0 and best_ask > 0) else 0.0
        midpoint = (best_bid + best_ask) / 2 if (best_bid > 0 and best_ask > 0) else 0.0

        return OrderBook(
            market=raw.get("market", ""),
            asset_id=raw.get("asset_id", ""),
            bids=bids,
            asks=asks,
            spread=spread,
            midpoint=midpoint,
            best_bid=best_bid,
            best_ask=best_ask,
            last_trade_price=float(raw.get("last_trade_price", 0)),
            min_order_size=float(raw.get("min_order_size", 0)),
            tick_size=float(raw.get("tick_size", 0.01)),
            timestamp=raw.get("timestamp", ""),
        )
