"""
Market Scanner — Find Betting Opportunities on Polymarket
==========================================================

This module pulls all active markets from Polymarket and flags opportunities
worth investigating. It does NOT make trading decisions -- it produces a
ranked list of "interesting" markets that a human (or future strategy module)
should look at more closely.

Four types of opportunities:

    1. Stale markets    — price is frozen, suggesting the market hasn't
                          incorporated recent information.
    2. Arb within market — multi-outcome market where prices don't sum to 1.00,
                          meaning you can buy all outcomes for less than the
                          guaranteed $1 payout (free money, minus fees).
    3. Wide spreads      — large gap between bid and ask. You might provide
                          liquidity or identify the "true" price in between.
    4. General scan      — markets that pass minimum volume/liquidity filters
                          but don't fall neatly into the categories above.
                          Flagged for manual review.

Usage:
    from bot.scanner.market_scanner import MarketScanner
    scanner = MarketScanner()
    results = scanner.scan_all()
    ranked = scanner.rank_opportunities(results)
    for r in ranked[:10]:
        print(r)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from bot.config import MIN_VOLUME, MIN_LIQUIDITY, STALE_HOURS
from bot.polymarket.client import PolymarketClient
from bot.polymarket.models import Market
from bot.scanner.models import ScanResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants (not in config.py because they are scanner-internal)
# ---------------------------------------------------------------------------

# A spread wider than this (as a fraction, e.g. 0.05 = 5%) is flagged.
WIDE_SPREAD_THRESHOLD: float = 0.05

# How close to a "round number" a price must be to count as stuck.
# 0.03 means prices within 3 cents of 0.10, 0.20, ... 0.90 are flagged.
ROUND_NUMBER_TOLERANCE: float = 0.03

# Round-number anchors that suggest a price is stuck on a default/lazy level.
ROUND_NUMBERS: list[float] = [
    0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90,
]

# If 24h volume is below this fraction of total volume, the market looks
# stale (nobody is actively trading it).
STALE_VOLUME_RATIO: float = 0.005

# An arb gap must be at least this large (in dollars per $1 of outcome
# tokens) to be worth flagging, because fees eat small gaps.
MIN_ARB_GAP: float = 0.02


# ---------------------------------------------------------------------------
# MarketScanner
# ---------------------------------------------------------------------------

class MarketScanner:
    """
    Scans Polymarket for betting opportunities.

    Args:
        client:        An existing PolymarketClient instance. If None, a new
                       one is created with default settings.
        min_volume:    Override the config MIN_VOLUME filter.
        min_liquidity: Override the config MIN_LIQUIDITY filter.
    """

    def __init__(
        self,
        client: Optional[PolymarketClient] = None,
        min_volume: float = MIN_VOLUME,
        min_liquidity: float = MIN_LIQUIDITY,
    ):
        self.client = client or PolymarketClient()
        self.min_volume = min_volume
        self.min_liquidity = min_liquidity

    # ------------------------------------------------------------------
    # 1. scan_all — fetch active markets, filter, return ScanResults
    # ------------------------------------------------------------------

    def scan_all(self, max_pages: int = 5) -> list[ScanResult]:
        """
        Fetch all active markets that pass volume/liquidity filters and
        return them as ScanResult objects with opportunity_type="manual_review".

        This is the broadest scan: every market that meets the minimums gets
        a result. Use the more targeted methods (find_stale_markets, etc.)
        or rank_opportunities() to narrow down.

        Args:
            max_pages: Maximum number of API pages to fetch (100 markets each).

        Returns:
            List of ScanResult objects, one per qualifying market.
        """
        now = _now_iso()
        logger.info(
            "scan_all: fetching active markets "
            f"(min_volume={self.min_volume}, min_liquidity={self.min_liquidity})"
        )

        try:
            markets = self.client.get_all_active_markets(
                min_volume=self.min_volume,
                min_liquidity=self.min_liquidity,
                max_pages=max_pages,
            )
        except Exception:
            logger.exception("scan_all: failed to fetch markets from Polymarket")
            return []

        logger.info(f"scan_all: {len(markets)} markets passed filters")

        results: list[ScanResult] = []
        for m in markets:
            results.append(ScanResult(
                market=m,
                opportunity_type="manual_review",
                score=0.0,
                details={
                    "volume": m.volume,
                    "volume_24h": m.volume_24h,
                    "liquidity": m.liquidity,
                    "num_outcomes": len(m.outcomes),
                },
                timestamp=now,
            ))

        return results

    # ------------------------------------------------------------------
    # 2. find_stale_markets — markets with frozen / under-traded prices
    # ------------------------------------------------------------------

    def find_stale_markets(
        self,
        markets: Optional[list[Market]] = None,
        max_pages: int = 5,
    ) -> list[ScanResult]:
        """
        Identify markets whose price appears stale / frozen.

        Indicators of staleness (we use proxies since we don't have tick-level
        historical data):

            a) Low recent volume — 24h volume is a tiny fraction of total
               volume, meaning almost nobody has traded recently.

            b) Price near a round number — prices like 0.50, 0.90, etc.
               often mean the market settled on a "default" and nobody has
               bothered to update it.

            c) Near expiry with a wide spread — if the market is about to
               close but the bid-ask spread is still wide, the market hasn't
               priced in recent information.

        A market is flagged as stale if it triggers ANY of these indicators.

        Args:
            markets: Pre-fetched list of markets. If None, fetches fresh data.
            max_pages: Pages to fetch if markets is None.

        Returns:
            List of ScanResults with opportunity_type="stale".
        """
        if markets is None:
            markets = self._fetch_filtered_markets(max_pages)

        now = _now_iso()
        results: list[ScanResult] = []

        for m in markets:
            staleness_signals: dict[str, object] = {}

            # (a) Low volume ratio
            volume_ratio = _safe_ratio(m.volume_24h, m.volume)
            if volume_ratio is not None and volume_ratio < STALE_VOLUME_RATIO:
                staleness_signals["low_volume_ratio"] = True
                staleness_signals["volume_ratio_24h"] = round(volume_ratio, 6)

            # (b) Price near round number
            yes_price = m.yes_price
            if yes_price is not None and _near_round_number(yes_price):
                staleness_signals["price_near_round"] = True
                staleness_signals["yes_price"] = yes_price

            # (c) Near expiry with wide spread
            hours_left = _hours_until(m.end_date)
            spread = _market_spread(m)
            if hours_left is not None and hours_left < STALE_HOURS * 2:
                staleness_signals["hours_to_expiry"] = round(hours_left, 1)
                if spread is not None and spread > WIDE_SPREAD_THRESHOLD:
                    staleness_signals["near_expiry_wide_spread"] = True
                    staleness_signals["spread"] = round(spread, 4)

            if staleness_signals:
                # Score: more signals = more stale. Normalize to 0-1.
                signal_count = sum(
                    1 for k in staleness_signals
                    if k in (
                        "low_volume_ratio",
                        "price_near_round",
                        "near_expiry_wide_spread",
                    )
                )
                score = min(signal_count / 3.0, 1.0)

                results.append(ScanResult(
                    market=m,
                    opportunity_type="stale",
                    score=score,
                    details=staleness_signals,
                    timestamp=now,
                ))

        logger.info(f"find_stale_markets: {len(results)} stale markets found")
        return results

    # ------------------------------------------------------------------
    # 3. find_arb_within_market — multi-outcome price sums != 1.00
    # ------------------------------------------------------------------

    def find_arb_within_market(
        self,
        markets: Optional[list[Market]] = None,
        max_pages: int = 5,
    ) -> list[ScanResult]:
        """
        For multi-outcome markets, check if outcome prices sum to != 1.00.

        How the arb works:
            - A market with outcomes A, B, C should have prices summing to 1.00
              because exactly one outcome will pay out $1.
            - If the sum < 1.00, you can buy ALL outcomes for less than $1 and
              are guaranteed to receive $1 when the market resolves. That
              difference (minus fees) is risk-free profit.
            - If the sum > 1.00, the market is "overpriced" overall. You could
              theoretically sell all outcomes, but that requires shorting which
              is harder on Polymarket.

        We only flag opportunities where the gap exceeds MIN_ARB_GAP to
        account for transaction fees (~2% per side on Polymarket).

        Args:
            markets: Pre-fetched list of markets. If None, fetches fresh data.
            max_pages: Pages to fetch if markets is None.

        Returns:
            List of ScanResults with opportunity_type="arb".
        """
        if markets is None:
            markets = self._fetch_filtered_markets(max_pages)

        now = _now_iso()
        results: list[ScanResult] = []

        for m in markets:
            if len(m.outcome_prices) < 2:
                continue

            price_sum = sum(m.outcome_prices)

            # Only flag if the gap is meaningful.
            # gap > 0 means sum < 1 (buy-all arb).
            # gap < 0 means sum > 1 (sell-all arb, harder to execute).
            gap = 1.0 - price_sum

            if abs(gap) < MIN_ARB_GAP:
                continue

            # Profit per dollar of outlay (for buy-all arb, gap > 0):
            #   You spend price_sum, receive 1.00. Profit = gap.
            #   Profit rate = gap / price_sum.
            if price_sum > 0:
                profit_per_dollar = gap / price_sum
            else:
                profit_per_dollar = 0.0

            # Score: scale the gap size. A 6% gap is excellent; normalize so
            # a 10%+ gap scores 1.0.
            score = min(abs(gap) / 0.10, 1.0)

            direction = "buy_all" if gap > 0 else "sell_all"

            results.append(ScanResult(
                market=m,
                opportunity_type="arb",
                score=score,
                details={
                    "price_sum": round(price_sum, 4),
                    "gap": round(gap, 4),
                    "abs_gap": round(abs(gap), 4),
                    "direction": direction,
                    "profit_per_dollar": round(profit_per_dollar, 4),
                    "num_outcomes": len(m.outcome_prices),
                    "outcome_prices": [round(p, 4) for p in m.outcome_prices],
                },
                timestamp=now,
            ))

        logger.info(f"find_arb_within_market: {len(results)} arb opportunities found")
        return results

    # ------------------------------------------------------------------
    # 4. find_wide_spread_markets — bid-ask spread > threshold
    # ------------------------------------------------------------------

    def find_wide_spread_markets(
        self,
        markets: Optional[list[Market]] = None,
        max_pages: int = 5,
        threshold: float = WIDE_SPREAD_THRESHOLD,
    ) -> list[ScanResult]:
        """
        Find markets where the bid-ask spread is wider than the threshold.

        Why this matters:
            - A wide spread means market makers aren't competing. There may
              be an opportunity to provide liquidity (place limit orders
              inside the spread) and earn the spread as profit.
            - Alternatively, the "true" price is somewhere between bid and
              ask. If you have a view, you can get in at a better price
              than the current midpoint.

        We use the spread data from the Gamma API (already on the Market
        object). For more precision, you could fetch the full order book
        from the CLOB API, but that's one API call per market and would
        be very slow for a broad scan.

        Args:
            markets:   Pre-fetched list of markets. If None, fetches fresh data.
            max_pages: Pages to fetch if markets is None.
            threshold: Minimum spread to flag (default 0.05 = 5 cents).

        Returns:
            List of ScanResults with opportunity_type="wide_spread".
        """
        if markets is None:
            markets = self._fetch_filtered_markets(max_pages)

        now = _now_iso()
        results: list[ScanResult] = []

        for m in markets:
            spread = _market_spread(m)
            if spread is None or spread <= threshold:
                continue

            # Relative spread: spread as a fraction of the midpoint price.
            # A 5-cent spread on a 50-cent market (10%) is worse than
            # a 5-cent spread on a 90-cent market (5.5%).
            midpoint = _market_midpoint(m)
            relative_spread = (spread / midpoint) if midpoint and midpoint > 0 else 0.0

            # Score: wider spread = higher score (more opportunity).
            # Normalize so that a 15%+ spread scores 1.0.
            score = min(spread / 0.15, 1.0)

            results.append(ScanResult(
                market=m,
                opportunity_type="wide_spread",
                score=score,
                details={
                    "spread": round(spread, 4),
                    "relative_spread": round(relative_spread, 4),
                    "best_bid": m.best_bid,
                    "best_ask": m.best_ask,
                    "midpoint": round(midpoint, 4) if midpoint else None,
                    "liquidity": m.liquidity,
                },
                timestamp=now,
            ))

        logger.info(
            f"find_wide_spread_markets: {len(results)} wide-spread markets found"
        )
        return results

    # ------------------------------------------------------------------
    # 5. rank_opportunities — sort results by attractiveness
    # ------------------------------------------------------------------

    def rank_opportunities(
        self,
        results: list[ScanResult],
    ) -> list[ScanResult]:
        """
        Rank scan results by overall attractiveness.

        The ranking considers multiple factors beyond the raw score:

            - Edge size (the score already captures this, but we boost arbs).
            - Liquidity (can we actually execute a trade?).
            - Time to resolution (sooner = faster capital turnover).
            - Volume (more volume = more reliable price discovery).

        Each factor is normalized to [0, 1] and combined with weights.
        The final composite score replaces the existing score on each result.

        Args:
            results: List of ScanResult objects to rank.

        Returns:
            The same list, sorted descending by composite score.
            Each result's .score is updated to the composite value.
        """
        if not results:
            return results

        # Collect raw values for normalization.
        liquidities = [r.market.liquidity for r in results]
        volumes = [r.market.volume for r in results]
        max_liquidity = max(liquidities) if liquidities else 1.0
        max_volume = max(volumes) if volumes else 1.0

        # Weights for the composite score.
        # Time is weighted heavily because fast-resolving markets let us
        # reinvest capital sooner (compounding effect). A 10% edge over
        # 7 days is ~521% annualized; the same edge over 90 days is ~40%.
        W_EDGE = 0.35        # How big is the mispricing?
        W_LIQUIDITY = 0.20   # Can we actually trade?
        W_TIME = 0.30        # How soon does it resolve? (high weight!)
        W_VOLUME = 0.15      # Is the price discovery reliable?

        for r in results:
            # Factor 1: Edge / opportunity quality (use the type-specific score).
            edge_score = r.score

            # Factor 2: Liquidity (normalized against the most liquid in batch).
            liq_score = (
                r.market.liquidity / max_liquidity
                if max_liquidity > 0 else 0.0
            )

            # Factor 3: Time to resolution.
            hours_left = _hours_until(r.market.end_date)
            if hours_left is not None:
                # Markets resolving within 24h score highest. Beyond 720h
                # (30 days) they score near 0. Use inverse scaling.
                if hours_left <= 0:
                    time_score = 0.0  # Already expired, not useful.
                elif hours_left <= 24:
                    time_score = 1.0
                elif hours_left <= 720:
                    # Linear decay from 1.0 at 24h to 0.1 at 720h.
                    time_score = 1.0 - 0.9 * ((hours_left - 24) / (720 - 24))
                else:
                    time_score = 0.1
            else:
                # Unknown end date — assign a middling score.
                time_score = 0.3

            # Factor 4: Volume (normalized).
            vol_score = (
                r.market.volume / max_volume
                if max_volume > 0 else 0.0
            )

            composite = (
                W_EDGE * edge_score
                + W_LIQUIDITY * liq_score
                + W_TIME * time_score
                + W_VOLUME * vol_score
            )

            # Clamp to [0, 1].
            r.score = max(0.0, min(1.0, composite))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Full pipeline: scan + categorize + rank
    # ------------------------------------------------------------------

    def run_full_scan(self, max_pages: int = 5) -> list[ScanResult]:
        """
        Convenience method that runs all scan types, deduplicates, and ranks.

        Steps:
            1. Fetch all active, filtered markets once.
            2. Run each detector (stale, arb, wide_spread) on that list.
            3. Merge results, keeping the highest-scoring entry per market
               if a market appears in multiple categories.
            4. Rank everything and return sorted.

        Returns:
            Ranked list of ScanResult objects.
        """
        markets = self._fetch_filtered_markets(max_pages)
        if not markets:
            logger.warning("run_full_scan: no markets passed filters")
            return []

        logger.info(f"run_full_scan: analyzing {len(markets)} markets")

        # Run all detectors on the same market list (avoids redundant API calls).
        stale = self.find_stale_markets(markets=markets)
        arbs = self.find_arb_within_market(markets=markets)
        wide = self.find_wide_spread_markets(markets=markets)

        # Merge: keep the highest-scoring result per market condition_id.
        best_by_market: dict[str, ScanResult] = {}
        for result in stale + arbs + wide:
            cid = result.market.condition_id
            if cid not in best_by_market or result.score > best_by_market[cid].score:
                best_by_market[cid] = result

        merged = list(best_by_market.values())
        logger.info(
            f"run_full_scan: {len(merged)} unique opportunities "
            f"(stale={len(stale)}, arb={len(arbs)}, wide_spread={len(wide)})"
        )

        return self.rank_opportunities(merged)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_filtered_markets(self, max_pages: int = 5) -> list[Market]:
        """Fetch active markets with the configured filters. Returns empty
        list on API error instead of crashing."""
        try:
            return self.client.get_all_active_markets(
                min_volume=self.min_volume,
                min_liquidity=self.min_liquidity,
                max_pages=max_pages,
            )
        except Exception:
            logger.exception("Failed to fetch markets from Polymarket")
            return []


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return numerator/denominator, or None if denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def _near_round_number(price: float) -> bool:
    """True if price is within ROUND_NUMBER_TOLERANCE of a common round number."""
    for rn in ROUND_NUMBERS:
        if abs(price - rn) <= ROUND_NUMBER_TOLERANCE:
            return True
    return False


def _market_spread(market: Market) -> Optional[float]:
    """
    Get the bid-ask spread for a market.

    Tries market.spread first (populated by Gamma API). Falls back to
    computing from best_bid / best_ask. Returns None if data is missing.
    """
    if market.spread > 0:
        return market.spread
    if market.best_bid > 0 and market.best_ask > 0:
        return market.best_ask - market.best_bid
    return None


def _market_midpoint(market: Market) -> Optional[float]:
    """
    Get the midpoint price for a market.

    Midpoint = (best_bid + best_ask) / 2. Returns None if data is missing.
    """
    if market.best_bid > 0 and market.best_ask > 0:
        return (market.best_bid + market.best_ask) / 2.0
    # Fall back to first outcome price if available.
    if market.yes_price is not None and market.yes_price > 0:
        return market.yes_price
    return None


def _hours_until(end_date_str: str) -> Optional[float]:
    """
    Parse an ISO 8601 date string and return hours from now until that date.

    Returns None if the string is empty or unparseable.
    """
    if not end_date_str:
        return None

    try:
        # Handle various ISO formats the API might return.
        # Strip trailing 'Z' and replace with +00:00 for fromisoformat().
        cleaned = end_date_str.replace("Z", "+00:00")
        end_dt = datetime.fromisoformat(cleaned)

        # If the parsed datetime is naive, assume UTC.
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta = end_dt - now
        return delta.total_seconds() / 3600.0
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_results(results: list[ScanResult], title: str, limit: int = 20) -> None:
    """Pretty-print a list of scan results to the console."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  {len(results)} result(s) found")
    print(f"{'=' * 70}")

    for i, r in enumerate(results[:limit], 1):
        question = r.market.question[:65]
        print(f"\n  #{i}  [{r.opportunity_type.upper()}]  score={r.score:.3f}")
        print(f"      {question}")
        print(f"      volume=${r.market.volume:,.0f}  "
              f"liq=${r.market.liquidity:,.0f}  "
              f"24h=${r.market.volume_24h:,.0f}")

        # Type-specific details.
        if r.opportunity_type == "arb":
            d = r.details
            print(f"      prices sum={d.get('price_sum', '?')}  "
                  f"gap={d.get('gap', '?')}  "
                  f"direction={d.get('direction', '?')}")
        elif r.opportunity_type == "wide_spread":
            d = r.details
            print(f"      spread={d.get('spread', '?')}  "
                  f"bid={d.get('best_bid', '?')}  "
                  f"ask={d.get('best_ask', '?')}")
        elif r.opportunity_type == "stale":
            d = r.details
            flags = []
            if d.get("low_volume_ratio"):
                flags.append(f"vol_ratio={d.get('volume_ratio_24h', '?')}")
            if d.get("price_near_round"):
                flags.append(f"round_price={d.get('yes_price', '?')}")
            if d.get("near_expiry_wide_spread"):
                flags.append(
                    f"expiry_spread(hrs={d.get('hours_to_expiry', '?')}, "
                    f"spread={d.get('spread', '?')})"
                )
            print(f"      signals: {', '.join(flags) if flags else 'none'}")

        if r.market.end_date:
            hours = _hours_until(r.market.end_date)
            if hours is not None:
                if hours <= 0:
                    print(f"      expires: EXPIRED")
                elif hours < 24:
                    print(f"      expires: {hours:.1f} hours")
                else:
                    print(f"      expires: {hours / 24:.1f} days")

    if len(results) > limit:
        print(f"\n  ... and {len(results) - limit} more results not shown.")
    print()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Polymarket Scanner")
    print("=" * 40)
    print(f"Filters: min_volume=${MIN_VOLUME}, min_liquidity=${MIN_LIQUIDITY}")
    print(f"Stale threshold: {STALE_HOURS}h")
    print(f"Wide spread threshold: {WIDE_SPREAD_THRESHOLD * 100:.0f}%")
    print()

    scanner = MarketScanner()

    # Determine which scan to run.
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "full":
        print("Running full scan (stale + arb + wide spread) ...")
        results = scanner.run_full_scan(max_pages=3)
        _print_results(results, "FULL SCAN — Ranked Opportunities")

    elif mode == "stale":
        print("Scanning for stale markets ...")
        results = scanner.find_stale_markets(max_pages=3)
        _print_results(results, "STALE MARKETS")

    elif mode == "arb":
        print("Scanning for arbitrage opportunities ...")
        results = scanner.find_arb_within_market(max_pages=3)
        _print_results(results, "ARBITRAGE OPPORTUNITIES")

    elif mode == "spread":
        print("Scanning for wide-spread markets ...")
        results = scanner.find_wide_spread_markets(max_pages=3)
        _print_results(results, "WIDE SPREAD MARKETS")

    elif mode == "all":
        print("Fetching all qualifying markets ...")
        results = scanner.scan_all(max_pages=3)
        _print_results(results, "ALL QUALIFYING MARKETS")

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python -m bot.scanner.market_scanner [full|stale|arb|spread|all]")
        sys.exit(1)
