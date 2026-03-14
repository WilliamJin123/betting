"""
Arb Detector — Scan + Optimize Multi-Outcome Arbitrage
=======================================================

Combines the scanner's arb detection with the Frank-Wolfe optimizer.
The scanner finds WHICH markets have an arb opportunity. This module
then figures out HOW MUCH to allocate to each outcome, accounting for
order book depth, slippage, and fees.

Usage:
    from bot.arb.detector import ArbDetector
    from bot.polymarket.client import PolymarketClient

    client = PolymarketClient()
    detector = ArbDetector(client, budget=10.0)
    opportunities = detector.scan_and_optimize()

    for opp in opportunities:
        opp.allocation.print_report()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from bot.polymarket.client import PolymarketClient
from bot.polymarket.models import Market, OrderBook
from bot.arb.optimizer import ArbOptimizer, ArbAllocation

logger = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity with an optimized allocation."""

    market: Market
    price_sum: float          # sum of outcome prices
    gap: float                # 1.0 - price_sum (positive = buy-all arb)
    allocation: ArbAllocation # optimizer result


class ArbDetector:
    """
    Scans for multi-outcome arb opportunities and optimizes allocations.

    Workflow:
        1. Fetch active markets from Polymarket.
        2. Find multi-outcome markets where prices sum to < 1.00.
        3. For each, fetch order books for all outcomes.
        4. Run the Frank-Wolfe optimizer.
        5. Return sorted by net profit descending.
    """

    def __init__(
        self,
        client: PolymarketClient,
        budget: float = 10.0,
        fee_rate: float = 0.02,
        max_iterations: int = 100,
    ):
        self.client = client
        self.budget = budget
        self.fee_rate = fee_rate
        self.max_iterations = max_iterations

    def scan_and_optimize(
        self,
        min_gap: float = 0.02,
        fetch_order_books: bool = True,
        max_pages: int = 5,
        min_volume: float = 1000,
        min_liquidity: float = 500,
    ) -> list[ArbOpportunity]:
        """
        Full pipeline: scan for arb opportunities and optimize each one.

        Args:
            min_gap:          Minimum price gap (1 - sum_prices) to consider.
                              Must exceed fees to be profitable. Default 0.02.
            fetch_order_books: Whether to fetch real order books from the CLOB
                              API. If False, uses simple price-only model.
            max_pages:        Max pages of markets to fetch from Gamma API.
            min_volume:       Minimum total volume filter.
            min_liquidity:    Minimum liquidity filter.

        Returns:
            List of ArbOpportunity objects, sorted by net profit descending.
        """
        # Step 1: Fetch active markets
        logger.info("Fetching active markets...")
        try:
            markets = self.client.get_all_active_markets(
                min_volume=min_volume,
                min_liquidity=min_liquidity,
                max_pages=max_pages,
            )
        except Exception:
            logger.exception("Failed to fetch markets")
            return []

        logger.info(f"Fetched {len(markets)} markets, scanning for arbs...")

        # Step 2: Find arb candidates
        candidates: list[tuple[Market, float, float]] = []
        for m in markets:
            if len(m.outcome_prices) < 2:
                continue

            # Filter out markets with invalid prices
            if any(p <= 0 or p >= 1.0 for p in m.outcome_prices):
                continue

            price_sum = sum(m.outcome_prices)
            gap = 1.0 - price_sum

            if gap >= min_gap:
                candidates.append((m, price_sum, gap))

        logger.info(f"Found {len(candidates)} arb candidates (gap >= {min_gap})")

        if not candidates:
            return []

        # Sort candidates by gap descending (best arb first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Step 3 & 4: For each candidate, fetch order books and optimize
        opportunities: list[ArbOpportunity] = []

        for market, price_sum, gap in candidates:
            logger.info(
                f"Optimizing: {market.question[:60]}... "
                f"(gap={gap:.4f}, {len(market.tokens)} outcomes)"
            )

            # Fetch order books if requested
            order_books: Optional[list[OrderBook]] = None
            if fetch_order_books and market.tokens:
                order_books = self._fetch_order_books(market)
                if order_books is None:
                    logger.warning(
                        f"Failed to fetch order books for {market.question[:40]}. "
                        "Using price-only model."
                    )

            # Run optimizer
            optimizer = ArbOptimizer(
                budget=self.budget,
                fee_rate=self.fee_rate,
                max_iterations=self.max_iterations,
            )

            try:
                allocation = optimizer.optimize(
                    outcome_prices=market.outcome_prices,
                    order_books=order_books,
                )
            except Exception:
                logger.exception(
                    f"Optimizer failed for {market.question[:40]}"
                )
                continue

            # Only include if profitable
            if allocation.net_profit > 0:
                opportunities.append(ArbOpportunity(
                    market=market,
                    price_sum=price_sum,
                    gap=gap,
                    allocation=allocation,
                ))
            else:
                logger.debug(
                    f"Skipping {market.question[:40]}: "
                    f"net_profit={allocation.net_profit:.4f} (not profitable after fees)"
                )

        # Step 5: Sort by net profit descending
        opportunities.sort(
            key=lambda o: o.allocation.net_profit,
            reverse=True,
        )

        logger.info(
            f"Scan complete: {len(opportunities)} profitable arb opportunities"
        )
        return opportunities

    def _fetch_order_books(self, market: Market) -> Optional[list[OrderBook]]:
        """
        Fetch order books for all outcome tokens in a market.

        Returns None if any fetch fails (we need ALL outcomes for the
        optimizer to work correctly).
        """
        books: list[OrderBook] = []

        for token in market.tokens:
            if not token.token_id:
                return None
            try:
                book = self.client.get_order_book(token.token_id)
                books.append(book)
            except Exception:
                logger.warning(
                    f"Failed to fetch order book for token {token.token_id} "
                    f"(outcome: {token.outcome})"
                )
                return None

        return books if len(books) == len(market.tokens) else None


def print_opportunities(opportunities: list[ArbOpportunity]) -> None:
    """Print a summary of all arb opportunities found."""
    if not opportunities:
        print("\n  No profitable arb opportunities found.")
        print("  This is normal -- efficient markets rarely have arbs.\n")
        return

    print()
    print("=" * 70)
    print(f"  Found {len(opportunities)} Profitable Arb Opportunity(ies)")
    print("=" * 70)

    for i, opp in enumerate(opportunities, 1):
        a = opp.allocation
        print(f"\n  #{i}  {opp.market.question[:60]}")
        print(f"      Outcomes:     {len(opp.market.outcome_prices)}")
        print(f"      Prices:       {[round(p, 4) for p in opp.market.outcome_prices]}")
        print(f"      Price sum:    {opp.price_sum:.4f}")
        print(f"      Gap:          {opp.gap:.4f} ({opp.gap * 100:.2f}%)")
        print(f"      Total cost:   ${a.total_cost:.4f}")
        print(f"      Min shares:   {a.min_shares:.4f}")
        print(f"      Net profit:   ${a.net_profit:.4f}")
        print(f"      ROI:          {a.roi:.2%}")
        print(f"      Converged:    {a.converged} ({a.iterations} iters)")

        # Per-outcome breakdown
        print(f"      {'Outcome':<10} {'Price':>7} {'Alloc':>9} {'Shares':>9}")
        print(f"      {'-' * 38}")
        for j, entry in enumerate(a.per_outcome):
            outcome_label = (
                opp.market.outcomes[j]
                if j < len(opp.market.outcomes)
                else f"#{j}"
            )
            outcome_label = outcome_label[:10]
            print(
                f"      {outcome_label:<10} "
                f"{entry['price']:>7.4f} "
                f"${entry['allocation']:>8.4f} "
                f"{entry['shares']:>9.4f}"
            )

    print()
    print("=" * 70)
    print()
