"""
Market Resolver — Checks if open trades have resolved on Polymarket.
====================================================================

This module periodically checks whether markets we have open bets on have
been resolved. When a market resolves, it updates our trade log with the
outcome (WIN/LOSS) and calculates the P&L.

How Polymarket resolution works:
    A market is resolved when:
        1. The market's ``active`` flag becomes False
        2. One of the outcome tokens has ``winner=True``

    We detect this by fetching the market from the CLOB API (which includes
    the ``winner`` field on tokens) and checking both conditions.

Usage:
    from bot.polymarket.client import PolymarketClient
    from bot.tracker.trade_logger import TradeLogger
    from bot.tracker.resolver import MarketResolver

    client = PolymarketClient()
    logger = TradeLogger()
    resolver = MarketResolver(client, logger)

    # Check all open trades and resolve any that are settled
    newly_resolved = resolver.check_resolutions()
"""

from __future__ import annotations

import logging
from typing import Any

from bot.polymarket.client import PolymarketClient
from bot.tracker.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


class MarketResolver:
    """Checks Polymarket for resolved markets and updates the trade log.

    Parameters
    ----------
    client : PolymarketClient
        API client used to fetch market status from Polymarket.
    trade_logger : TradeLogger
        The trade ledger where resolved outcomes will be recorded.
    """

    def __init__(self, client: PolymarketClient, trade_logger: TradeLogger) -> None:
        self.client = client
        self.trade_logger = trade_logger

    def check_resolutions(self) -> list[dict[str, Any]]:
        """Check all open trades and resolve any whose markets have settled.

        For each open trade, this method:
            1. Fetches the market from the CLOB API (which has the ``winner``
               field on tokens).
            2. Checks if the market is no longer active.
            3. Checks if any token has ``winner=True``.
            4. If resolved, determines the YES/NO outcome and calls
               ``trade_logger.resolve_trade()``.

        Returns
        -------
        list[dict]
            All trade records that were newly resolved in this run.
            Empty list if nothing new was resolved.
        """
        open_trades = self.trade_logger.get_open_trades()
        if not open_trades:
            logger.info("No open trades to check.")
            return []

        # Deduplicate market IDs -- multiple bets can be on the same market
        market_ids: set[str] = set()
        for trade in open_trades:
            market_ids.add(trade["market_id"])

        logger.info(
            "Checking %d open trades across %d markets for resolution.",
            len(open_trades),
            len(market_ids),
        )

        all_newly_resolved: list[dict[str, Any]] = []

        for market_id in market_ids:
            try:
                resolved_trades = self._check_single_market(market_id)
                all_newly_resolved.extend(resolved_trades)
            except Exception:
                # Log the error but keep checking other markets.
                # A single API failure shouldn't block everything.
                logger.exception(
                    "Error checking resolution for market %s", market_id
                )

        if all_newly_resolved:
            logger.info(
                "Resolved %d trades in this run.", len(all_newly_resolved)
            )
        else:
            logger.info("No markets have resolved since last check.")

        return all_newly_resolved

    def _check_single_market(self, market_id: str) -> list[dict[str, Any]]:
        """Check a single market for resolution and update trades if resolved.

        Parameters
        ----------
        market_id : str
            The condition_id of the market to check.

        Returns
        -------
        list[dict]
            Trade records that were resolved, or empty list if the market
            hasn't resolved yet.
        """
        market = self.client.get_clob_market(market_id)

        if market is None:
            # Market not found on CLOB. Try Gamma as a fallback.
            market = self.client.get_gamma_market_by_id(market_id)

        if market is None:
            logger.warning(
                "Market %s not found on either API. Skipping.", market_id
            )
            return []

        # A market is resolved when it's no longer active AND a winner exists
        if market.active:
            logger.debug("Market %s is still active.", market_id)
            return []

        # Find the winning token
        winning_outcome = self._find_winner(market)
        if winning_outcome is None:
            # Market is inactive but no winner set yet. This can happen
            # briefly during the resolution process.
            logger.debug(
                "Market %s is inactive but no winner found yet.", market_id
            )
            return []

        # Determine the boolean outcome: True = YES won, False = NO won
        # The first outcome/token is conventionally "Yes"
        yes_won = winning_outcome.upper() in ("YES", "Y")

        logger.info(
            "Market %s resolved: winner=%s (yes_won=%s)",
            market_id,
            winning_outcome,
            yes_won,
        )

        try:
            resolved_trades = self.trade_logger.resolve_trade(
                market_id=market_id,
                outcome=yes_won,
            )
            return resolved_trades
        except ValueError as e:
            # No open trades found for this market (already resolved)
            logger.debug("Resolve skipped for %s: %s", market_id, e)
            return []

    @staticmethod
    def _find_winner(market) -> str | None:
        """Determine which outcome won in a resolved market.

        Looks through the market's tokens for one with ``winner=True``.

        Returns
        -------
        str or None
            The outcome label of the winning token (e.g., "Yes" or "No"),
            or None if no winner has been set.
        """
        for token in market.tokens:
            if token.winner:
                return token.outcome
        return None
