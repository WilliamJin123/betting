"""
Backtesting Engine
===================

Runs a probability estimation strategy against historically resolved
Polymarket markets to measure accuracy, P&L, and calibration.

KEY ASSUMPTION (Option A — 50/50 proxy):
    The Gamma API returns *final* outcome prices for closed markets (0 or 1),
    NOT the pre-resolution market prices.  Since we don't have historical
    price snapshots, we assume the market was priced at 50/50 before
    resolution.  This is a rough approximation — it tests whether the
    strategy can do better than a coin flip, which is the minimum bar.

    A future improvement (Option B) would query historical CLOB data or
    use the Polymarket timeseries API to get actual pre-resolution prices.
"""

from __future__ import annotations

import math
import logging
from typing import Callable, Optional

from bot.backtester.data import ResolvedMarket
from bot.backtester.results import BacktestResult
from bot.sizing.kelly import kelly_bet
from bot.tracker.calibration import compute_calibration

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Run strategies against resolved markets and compute performance metrics.

    Args:
        strategy:           A callable that takes a ResolvedMarket and returns
                            a float (0-1) probability estimate for the YES
                            outcome, or None if the strategy has no opinion.
        bankroll:           Starting bankroll in USD.
        kelly_multiplier:   Fraction of Kelly to use (0.25 = quarter Kelly).
        min_edge:           Minimum edge required to place a bet.
        max_bet_fraction:   Maximum fraction of bankroll per bet.
        market_price_proxy: The assumed pre-resolution market price.
                            Default 0.50 (coin flip assumption).
    """

    def __init__(
        self,
        strategy: Callable[[ResolvedMarket], Optional[float]],
        bankroll: float = 10.0,
        kelly_multiplier: float = 0.25,
        min_edge: float = 0.05,
        max_bet_fraction: float = 0.15,
        market_price_proxy: float = 0.50,
    ):
        self.strategy = strategy
        self.initial_bankroll = bankroll
        self.kelly_multiplier = kelly_multiplier
        self.min_edge = min_edge
        self.max_bet_fraction = max_bet_fraction
        self.market_price_proxy = market_price_proxy

    def run(self, markets: list[ResolvedMarket]) -> BacktestResult:
        """
        Run the backtest across all provided resolved markets.

        For each market:
        1. Call strategy(market) to get p_estimated (probability of YES).
        2. Use self.market_price_proxy as the assumed pre-resolution price.
        3. Run Kelly sizing to determine bet side and size.
        4. Determine if the bet would have won based on the actual outcome.
        5. Calculate P&L and track running bankroll.

        Returns:
            BacktestResult with all performance statistics.
        """
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        max_drawdown = 0.0

        trades: list[dict] = []
        pnl_series: list[float] = []  # per-bet P&L for Sharpe calculation
        calibration_data: list[tuple[float, bool]] = []

        total_markets = 0  # markets strategy had an opinion on
        bets_placed = 0
        bets_skipped = 0
        wins = 0
        losses = 0
        total_edge = 0.0
        total_bet_size = 0.0

        for market in markets:
            # --- Step 1: Get strategy estimate ---
            try:
                p_estimated = self.strategy(market)
            except Exception as e:
                logger.debug("Strategy error on '%s': %s", market.question[:50], e)
                continue

            if p_estimated is None:
                continue  # strategy has no opinion

            # Clamp to valid range for Kelly
            p_estimated = max(0.01, min(0.99, p_estimated))
            total_markets += 1

            # --- Step 2: Determine market price proxy ---
            p_market = self.market_price_proxy

            # --- Step 3: Kelly sizing ---
            try:
                bet = kelly_bet(
                    p_true=p_estimated,
                    p_market=p_market,
                    bankroll=bankroll,
                    kelly_fraction_multiplier=self.kelly_multiplier,
                    max_bet_fraction=self.max_bet_fraction,
                    min_edge=self.min_edge,
                )
            except (ValueError, TypeError) as e:
                logger.debug("Kelly error on '%s': %s", market.question[:50], e)
                continue

            if bet["size"] <= 0:
                bets_skipped += 1
                continue

            bets_placed += 1
            side = bet["side"]
            size = bet["size"]
            total_edge += bet["abs_edge"]
            total_bet_size += size

            # --- Step 4: Determine if the bet would have won ---
            # The winning outcome is stored in market.winning_outcome.
            # If we bet YES, we win if winning_outcome is the first outcome.
            # If we bet NO, we win if winning_outcome is NOT the first outcome.
            yes_won = (
                len(market.final_prices) >= 1
                and market.final_prices[0] == 1.0
            )

            if side == "YES":
                won = yes_won
            else:
                won = not yes_won

            # --- Step 5: Calculate P&L ---
            # In a prediction market:
            #   - Buy YES at p_market: if YES wins, profit = (1 - p_market) * shares
            #                          if NO wins,  loss = p_market * shares
            #   - Buy NO at (1 - p_market): if NO wins, profit = p_market * shares
            #                                if YES wins, loss = (1 - p_market) * shares
            #
            # Since size is in USD and we buy at p_market:
            #   shares = size / entry_price
            #   If win: pnl = shares * (1 - entry_price) = size * (1 - entry_price) / entry_price
            #   If lose: pnl = -size (we lose the whole stake)
            if side == "YES":
                entry_price = p_market
            else:
                entry_price = 1.0 - p_market

            if won:
                # Profit: we paid entry_price per share, get $1 per share
                pnl = size * (1.0 - entry_price) / entry_price
                wins += 1
            else:
                # Loss: we lose our entire stake
                pnl = -size
                losses += 1

            bankroll += pnl
            pnl_series.append(pnl)

            # Track drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            # Track calibration: (estimated P(YES), did YES actually win?)
            calibration_data.append((p_estimated, yes_won))

            # Record trade
            trades.append({
                "question": market.question,
                "condition_id": market.condition_id,
                "side": side,
                "p_estimated": p_estimated,
                "p_market": p_market,
                "edge": bet["edge"],
                "abs_edge": bet["abs_edge"],
                "kelly_f": bet["kelly_f"],
                "adjusted_f": bet["adjusted_f"],
                "size": size,
                "entry_price": entry_price,
                "won": won,
                "pnl": round(pnl, 6),
                "bankroll_after": round(bankroll, 6),
                "winning_outcome": market.winning_outcome,
            })

        # --- Compute aggregate stats ---
        total_pnl = bankroll - self.initial_bankroll
        win_rate = wins / bets_placed if bets_placed > 0 else 0.0
        avg_edge = total_edge / bets_placed if bets_placed > 0 else 0.0
        avg_bet_size = total_bet_size / bets_placed if bets_placed > 0 else 0.0

        # Sharpe ratio (approximate annualized)
        sharpe = self._compute_sharpe(pnl_series)

        # Calibration
        calibration = {}
        if calibration_data:
            calibration = compute_calibration(calibration_data)

        return BacktestResult(
            total_markets=total_markets,
            bets_placed=bets_placed,
            bets_skipped=bets_skipped,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=round(total_pnl, 6),
            final_bankroll=round(bankroll, 6),
            max_drawdown=round(max_drawdown, 6),
            sharpe_ratio=round(sharpe, 4),
            avg_edge=round(avg_edge, 6),
            avg_bet_size=round(avg_bet_size, 6),
            calibration=calibration,
            trades=trades,
        )

    @staticmethod
    def _compute_sharpe(pnl_series: list[float], annualization_factor: float = 252.0) -> float:
        """
        Compute an approximate annualized Sharpe ratio from per-bet P&L.

        We treat each bet as one "period" and assume roughly 252 bets per year
        (one per trading day).  This is a rough approximation — real Sharpe
        calculations need actual time-based returns.

        Sharpe = (mean_return / std_return) * sqrt(annualization_factor)

        Returns 0.0 if there are fewer than 2 data points or std is zero.
        """
        if len(pnl_series) < 2:
            return 0.0

        mean_pnl = sum(pnl_series) / len(pnl_series)
        variance = sum((p - mean_pnl) ** 2 for p in pnl_series) / (len(pnl_series) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.0

        if std_pnl == 0:
            return 0.0

        return (mean_pnl / std_pnl) * math.sqrt(annualization_factor)
