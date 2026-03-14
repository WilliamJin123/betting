"""
Monte Carlo Simulator for Strategy Stress Testing
===================================================

Wraps the BacktestEngine to run thousands of randomized simulations,
producing a distribution of outcomes that answers questions like:

    - "What's my median expected bankroll after running this strategy?"
    - "What's the probability I go broke (ruin)?"
    - "What does the worst realistic scenario look like?"

Each simulation introduces controlled randomness:
    1. Shuffles the order of markets (different sequence = different drawdown path)
    2. Adds Gaussian noise to probability estimates (simulates estimation error)
    3. Randomly drops markets (simulates missed opportunities)

After N runs, the aggregated results give a statistical picture of strategy
robustness that a single backtest cannot provide.

Dependencies: stdlib only (random, math, statistics). No numpy.
"""

from __future__ import annotations

import random
import math
from statistics import mean, stdev, median, quantiles
from typing import Callable, Optional

from bot.backtester.data import ResolvedMarket
from bot.backtester.engine import BacktestEngine


class MonteCarloResult:
    """
    Aggregated results from Monte Carlo simulation.

    Stores the raw distribution of outcomes across all simulation runs
    and exposes summary statistics as properties.
    """

    def __init__(
        self,
        n_simulations: int,
        starting_bankroll: float,
        ruin_threshold: float,
        estimation_noise: float,
        dropout_rate: float,
        final_bankrolls: list[float],
        total_pnls: list[float],
        max_drawdowns: list[float],
        win_rates: list[float],
    ):
        self.n_simulations = n_simulations
        self.starting_bankroll = starting_bankroll
        self.ruin_threshold = ruin_threshold
        self.estimation_noise = estimation_noise
        self.dropout_rate = dropout_rate
        self.final_bankrolls = final_bankrolls
        self.total_pnls = total_pnls
        self.max_drawdowns = max_drawdowns
        self.win_rates = win_rates

    @property
    def probability_of_ruin(self) -> float:
        """Fraction of simulations where final bankroll fell below ruin threshold."""
        if not self.final_bankrolls:
            return 0.0
        ruin_count = sum(1 for b in self.final_bankrolls if b < self.ruin_threshold)
        return ruin_count / len(self.final_bankrolls)

    @property
    def median_final_bankroll(self) -> float:
        """50th percentile final bankroll."""
        if not self.final_bankrolls:
            return 0.0
        return median(self.final_bankrolls)

    @property
    def percentile_5(self) -> float:
        """5th percentile -- worst realistic outcome."""
        if len(self.final_bankrolls) < 2:
            return self.final_bankrolls[0] if self.final_bankrolls else 0.0
        # quantiles with n=20 gives 5% increments; index 0 is 5th percentile
        q = quantiles(self.final_bankrolls, n=20)
        return q[0]  # 5th percentile

    @property
    def percentile_25(self) -> float:
        """25th percentile."""
        if len(self.final_bankrolls) < 2:
            return self.final_bankrolls[0] if self.final_bankrolls else 0.0
        q = quantiles(self.final_bankrolls, n=4)
        return q[0]  # 25th percentile

    @property
    def percentile_75(self) -> float:
        """75th percentile."""
        if len(self.final_bankrolls) < 2:
            return self.final_bankrolls[0] if self.final_bankrolls else 0.0
        q = quantiles(self.final_bankrolls, n=4)
        return q[2]  # 75th percentile

    @property
    def percentile_95(self) -> float:
        """95th percentile -- best realistic outcome."""
        if len(self.final_bankrolls) < 2:
            return self.final_bankrolls[0] if self.final_bankrolls else 0.0
        q = quantiles(self.final_bankrolls, n=20)
        return q[-1]  # 95th percentile

    @property
    def expected_value(self) -> float:
        """Mean final bankroll across all simulations."""
        if not self.final_bankrolls:
            return 0.0
        return mean(self.final_bankrolls)

    @property
    def median_max_drawdown(self) -> float:
        """Median of the max drawdown across simulations."""
        if not self.max_drawdowns:
            return 0.0
        return median(self.max_drawdowns)

    @property
    def worst_max_drawdown(self) -> float:
        """Worst (highest) max drawdown seen across all simulations."""
        if not self.max_drawdowns:
            return 0.0
        return max(self.max_drawdowns)

    def _build_histogram(self, values: list[float], n_bins: int = 10, width: int = 40) -> list[str]:
        """
        Build a simple ASCII histogram.

        Returns a list of strings, one per line.
        """
        if not values:
            return ["  (no data)"]

        lo = min(values)
        hi = max(values)

        # Handle case where all values are identical
        if hi == lo:
            return [f"  All values = ${lo:.2f}  (N={len(values)})"]

        bin_width = (hi - lo) / n_bins
        bins = [0] * n_bins

        for v in values:
            idx = int((v - lo) / bin_width)
            # Edge case: value == hi goes in last bin
            if idx >= n_bins:
                idx = n_bins - 1
            bins[idx] += 1

        max_count = max(bins) if bins else 1
        lines = []

        for i in range(n_bins):
            bin_lo = lo + i * bin_width
            bin_hi = lo + (i + 1) * bin_width
            count = bins[i]
            bar_len = int((count / max_count) * width) if max_count > 0 else 0
            bar = "#" * bar_len
            label = f"  ${bin_lo:>7.2f} - ${bin_hi:>7.2f}"
            lines.append(f"{label} | {bar} ({count})")

        return lines

    def print_report(self) -> None:
        """Print a formatted Monte Carlo simulation report."""
        print()
        print("=" * 65)
        print("  MONTE CARLO SIMULATION REPORT")
        print("=" * 65)
        print()

        # --- Overview ---
        print("  --- Overview ---")
        print(f"  Simulations:         {self.n_simulations}")
        print(f"  Starting bankroll:   ${self.starting_bankroll:.2f}")
        print(f"  Estimation noise:    {self.estimation_noise:.2%} std dev")
        print(f"  Market dropout rate: {self.dropout_rate:.0%}")
        print(f"  Ruin threshold:      ${self.ruin_threshold:.2f}")
        print()

        # --- Outcome Distribution ---
        print("  --- Outcome Distribution (Final Bankroll) ---")
        print(f"  5th percentile:      ${self.percentile_5:.4f}  (worst realistic)")
        print(f"  25th percentile:     ${self.percentile_25:.4f}")
        print(f"  Median (50th):       ${self.median_final_bankroll:.4f}")
        print(f"  75th percentile:     ${self.percentile_75:.4f}")
        print(f"  95th percentile:     ${self.percentile_95:.4f}  (best realistic)")
        print(f"  Mean:                ${self.expected_value:.4f}")
        if len(self.final_bankrolls) >= 2:
            print(f"  Std dev:             ${stdev(self.final_bankrolls):.4f}")
        print(f"  Min:                 ${min(self.final_bankrolls):.4f}")
        print(f"  Max:                 ${max(self.final_bankrolls):.4f}")
        print()

        # --- P&L ---
        median_pnl = median(self.total_pnls) if self.total_pnls else 0.0
        mean_pnl = mean(self.total_pnls) if self.total_pnls else 0.0
        pnl_positive = sum(1 for p in self.total_pnls if p > 0)
        pnl_negative = sum(1 for p in self.total_pnls if p < 0)
        pnl_zero = sum(1 for p in self.total_pnls if p == 0)
        sign_med = "+" if median_pnl >= 0 else ""
        sign_mean = "+" if mean_pnl >= 0 else ""

        print("  --- P&L Distribution ---")
        print(f"  Median P&L:          {sign_med}${median_pnl:.4f}")
        print(f"  Mean P&L:            {sign_mean}${mean_pnl:.4f}")
        print(f"  Profitable sims:     {pnl_positive} / {self.n_simulations} ({pnl_positive / self.n_simulations:.1%})")
        print(f"  Losing sims:         {pnl_negative} / {self.n_simulations}")
        if pnl_zero > 0:
            print(f"  Break-even sims:     {pnl_zero}")
        print()

        # --- Risk Metrics ---
        print("  --- Risk Metrics ---")
        print(f"  Probability of ruin: {self.probability_of_ruin:.1%}  (bankroll < ${self.ruin_threshold:.2f})")
        print(f"  Median max drawdown: {self.median_max_drawdown:.2%}")
        print(f"  Worst max drawdown:  {self.worst_max_drawdown:.2%}")
        if len(self.max_drawdowns) >= 2:
            dd_95 = quantiles(self.max_drawdowns, n=20)[-1]
            print(f"  95th pctl drawdown:  {dd_95:.2%}")
        print()

        # --- Win Rate Distribution ---
        valid_wr = [w for w in self.win_rates if w > 0]
        if valid_wr:
            print("  --- Win Rate Distribution ---")
            print(f"  Median win rate:     {median(valid_wr):.1%}")
            print(f"  Min win rate:        {min(valid_wr):.1%}")
            print(f"  Max win rate:        {max(valid_wr):.1%}")
            print()

        # --- Histogram ---
        print("  --- Final Bankroll Histogram ---")
        hist_lines = self._build_histogram(self.final_bankrolls)
        for line in hist_lines:
            print(f"  {line}")
        print()

        # --- Verdict ---
        p_ruin = self.probability_of_ruin
        print("  --- Verdict ---")
        if p_ruin < 0.05:
            verdict = "SAFE TO DEPLOY"
            detail = f"P(ruin) = {p_ruin:.1%} < 5%"
        elif p_ruin <= 0.20:
            verdict = "HIGH RISK"
            detail = f"P(ruin) = {p_ruin:.1%} (5% - 20%)"
        else:
            verdict = "DO NOT DEPLOY"
            detail = f"P(ruin) = {p_ruin:.1%} > 20%"

        print(f"  {verdict}")
        print(f"  {detail}")
        print()

        # Additional context
        median_return = (self.median_final_bankroll - self.starting_bankroll) / self.starting_bankroll
        sign_ret = "+" if median_return >= 0 else ""
        print(f"  Median return: {sign_ret}{median_return:.2%} on ${self.starting_bankroll:.2f} bankroll")
        print()
        print("=" * 65)
        print()


class MonteCarloSimulator:
    """
    Runs N simulations of a strategy with randomized conditions to produce
    a distribution of outcomes.

    Each simulation:
    1. Shuffles the order of markets (different sequence = different drawdown path)
    2. Adds noise to the strategy's probability estimates (simulates estimation error)
    3. Optionally drops random markets (simulates missed opportunities)
    4. Runs the full backtest
    5. Records the final bankroll, max drawdown, P&L, etc.

    After N runs, we get:
    - Median final bankroll (50th percentile outcome)
    - 5th/95th percentile outcomes (confidence interval)
    - Probability of ruin (what fraction of runs went to $0 or below some threshold?)
    - Distribution of max drawdowns
    """

    def __init__(
        self,
        strategy: Callable[[ResolvedMarket], Optional[float]],
        markets: list[ResolvedMarket],
        n_simulations: int = 1000,
        bankroll: float = 10.0,
        kelly_multiplier: float = 0.25,
        estimation_noise: float = 0.05,
        dropout_rate: float = 0.1,
        ruin_threshold: float = 1.0,
    ):
        self.strategy = strategy
        self.markets = markets
        self.n_simulations = n_simulations
        self.bankroll = bankroll
        self.kelly_multiplier = kelly_multiplier
        self.estimation_noise = estimation_noise
        self.dropout_rate = dropout_rate
        self.ruin_threshold = ruin_threshold

    def _make_noisy_strategy(
        self,
    ) -> Callable[[ResolvedMarket], Optional[float]]:
        """
        Create a wrapper strategy that adds Gaussian noise to the base
        strategy's probability estimate.

        The noise simulates real-world estimation error: your model's
        P(YES) = 0.70 might actually be anywhere from 0.60 to 0.80
        depending on how good your model is.

        The noisy estimate is clamped to (0.01, 0.99) to keep it valid.
        """
        base_strategy = self.strategy
        noise_std = self.estimation_noise

        def noisy_strategy(market: ResolvedMarket) -> Optional[float]:
            result = base_strategy(market)
            if result is None:
                return None
            # Add Gaussian noise
            noisy = result + random.gauss(0, noise_std)
            # Clamp to valid probability range
            return max(0.01, min(0.99, noisy))

        return noisy_strategy

    def _filter_markets_with_dropout(
        self, markets: list[ResolvedMarket],
    ) -> list[ResolvedMarket]:
        """
        Randomly drop markets with probability `dropout_rate`.

        This simulates the real-world scenario where your bot misses
        some opportunities: API downtime, slow response, market already
        closed by the time you get to it, etc.
        """
        if self.dropout_rate <= 0:
            return markets
        return [m for m in markets if random.random() >= self.dropout_rate]

    def run(self) -> MonteCarloResult:
        """
        Run all simulations and return aggregated results.

        For each simulation:
          1. Shuffle markets (random order)
          2. Create a noisy wrapper around the strategy
          3. Randomly drop markets with probability dropout_rate
          4. Run BacktestEngine on the filtered+shuffled markets
          5. Record final_bankroll, total_pnl, max_drawdown, win_rate
        """
        final_bankrolls: list[float] = []
        total_pnls: list[float] = []
        max_drawdowns: list[float] = []
        win_rates: list[float] = []

        print(f"\n  Running {self.n_simulations} Monte Carlo simulations...")

        for i in range(self.n_simulations):
            # Progress indicator every 100 sims (or at 1, 10, 50 for small runs)
            sim_num = i + 1
            if sim_num == 1 or sim_num % 100 == 0 or sim_num == self.n_simulations:
                print(f"  Simulation {sim_num}/{self.n_simulations}...")

            # 1. Shuffle markets
            shuffled = list(self.markets)
            random.shuffle(shuffled)

            # 2. Create noisy strategy
            noisy_strategy = self._make_noisy_strategy()

            # 3. Dropout
            filtered = self._filter_markets_with_dropout(shuffled)

            # 4. Run backtest
            engine = BacktestEngine(
                strategy=noisy_strategy,
                bankroll=self.bankroll,
                kelly_multiplier=self.kelly_multiplier,
            )
            result = engine.run(filtered)

            # 5. Record results
            final_bankrolls.append(result.final_bankroll)
            total_pnls.append(result.total_pnl)
            max_drawdowns.append(result.max_drawdown)
            win_rates.append(result.win_rate)

        print(f"  All {self.n_simulations} simulations complete.")

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            starting_bankroll=self.bankroll,
            ruin_threshold=self.ruin_threshold,
            estimation_noise=self.estimation_noise,
            dropout_rate=self.dropout_rate,
            final_bankrolls=final_bankrolls,
            total_pnls=total_pnls,
            max_drawdowns=max_drawdowns,
            win_rates=win_rates,
        )
