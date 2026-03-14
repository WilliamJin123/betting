"""
Frank-Wolfe + Bregman Projection Optimizer for Multi-Outcome Arbitrage
======================================================================

When a multi-outcome Polymarket market (e.g., "Who wins the election?"
with candidates A, B, C, D) has prices summing to less than $1.00, you
can buy all outcomes and guarantee a profit because exactly one outcome
will pay out $1.00.

But HOW MUCH to buy of each outcome is not obvious:
    - Each outcome has different liquidity (thin order books)
    - Buying large amounts moves the price against you (slippage)
    - Fees (~2%) eat into the margin
    - You need the same number of shares of each outcome to redeem
      complete sets

This module uses a two-phase approach:

Phase 1 (Analytical):
    For the simple price-only model (infinite liquidity), the optimal
    allocation has a closed-form solution: allocate proportional to
    prices. If a_i = B * p_i / sum(p_j), then shares_i = a_i / p_i
    = B / sum(p_j) for all i -- perfectly balanced shares.

Phase 2 (Frank-Wolfe refinement with Bregman Projection):
    When order books introduce slippage, the analytical solution is
    no longer exact. We use coordinate-wise Frank-Wolfe with away-steps
    to refine the allocation. The Bregman Projection (KL-divergence)
    keeps the solution feasible while preserving relative proportions.

The Bregman Projection is preferable to Euclidean projection because
it preserves the multiplicative structure of the allocation. When one
outcome has price 0.05 and another has price 0.80, Euclidean projection
would distort the ratios, but KL-divergence projection scales them
proportionally.

No numpy/scipy required -- pure stdlib math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from bot.polymarket.models import OrderBook


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArbAllocation:
    """Result of the Frank-Wolfe optimization."""

    allocations: list[float]           # dollar amount per outcome
    shares_per_outcome: list[float]    # shares received per outcome
    min_shares: float                  # minimum across outcomes (redeemable sets)
    total_cost: float                  # sum of allocations
    gross_profit: float                # min_shares * 1.0 - total_cost
    fees: float                        # estimated fees
    net_profit: float                  # gross - fees
    roi: float                         # net_profit / total_cost (0 if no cost)
    iterations: int                    # how many Frank-Wolfe iterations ran
    converged: bool                    # did it converge within tolerance?
    per_outcome: list[dict] = field(default_factory=list)
    # [{outcome_index, price, allocation, shares}, ...]

    def print_report(self) -> None:
        """Print a formatted allocation report to stdout."""
        print()
        print("=" * 60)
        print("  Arb Allocation Report (Frank-Wolfe Optimizer)")
        print("=" * 60)
        print(f"  Total cost:    ${self.total_cost:.4f}")
        print(f"  Min shares:    {self.min_shares:.4f}")
        print(f"  Gross profit:  ${self.gross_profit:.4f}")
        print(f"  Fees:          ${self.fees:.4f}")
        print(f"  Net profit:    ${self.net_profit:.4f}")
        if self.total_cost > 0:
            print(f"  ROI:           {self.roi:.2%}")
        else:
            print(f"  ROI:           N/A (no cost)")
        print(f"  Iterations:    {self.iterations}")
        print(f"  Converged:     {self.converged}")
        print()
        print(f"  {'#':<4} {'Price':>7} {'Alloc $':>9} {'Shares':>9}")
        print(f"  {'-' * 33}")
        for entry in self.per_outcome:
            idx = entry.get("outcome_index", "?")
            price = entry.get("price", 0.0)
            alloc = entry.get("allocation", 0.0)
            shares = entry.get("shares", 0.0)
            print(f"  {idx:<4} {price:>7.4f} {alloc:>9.4f} {shares:>9.4f}")
        print("=" * 60)
        print()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class ArbOptimizer:
    """
    Optimizes allocation across multiple outcomes in an arbitrage
    opportunity using the Adaptive Frank-Wolfe algorithm with Bregman
    Projection.

    The problem:
        Given a market with K outcomes and budget B, find the dollar
        allocation [a_1, a_2, ..., a_K] that maximizes guaranteed profit:

            profit = min(shares_1, ..., shares_K) * $1.00 - sum(a_i)

        where shares_i depends on allocation and the order book depth.

    Constraints:
        - sum(a_i) <= budget
        - a_i >= 0 for all i
        - shares purchased account for slippage from order book
    """

    def __init__(
        self,
        budget: float,
        fee_rate: float = 0.02,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        if budget <= 0:
            raise ValueError(f"Budget must be positive, got {budget}")
        self.budget = budget
        self.fee_rate = fee_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        outcome_prices: list[float],
        order_books: Optional[list[OrderBook]] = None,
    ) -> ArbAllocation:
        """
        Find optimal allocation across outcomes.

        Two-phase approach:

        Phase 1 (Analytical):
            Compute the closed-form optimal for the price-only model.
            Allocate proportional to price: a_i = B * p_i / sum(p_j).
            This equalizes shares across all outcomes.

        Phase 2 (Frank-Wolfe refinement):
            If order books are provided, refine Phase 1 result using
            coordinate-wise gradient ascent. At each step:
            a. Compute gradient via finite differences
            b. Move budget from the outcome with the worst marginal
               return to the one with the best marginal return
            c. Apply Bregman Projection (proportional scaling)
            d. Check convergence

        Then search over budget utilization levels (10%-100%) to find
        the optimal scale, since with slippage, spending less may
        yield higher profit.

        Returns:
            ArbAllocation with per-outcome dollar amounts and profit.
        """
        k = len(outcome_prices)

        # --- Edge cases ---
        if k < 2:
            return self._empty_allocation(outcome_prices, 0, True)

        # Validate prices
        for i, p in enumerate(outcome_prices):
            if p <= 0 or p >= 1.0:
                return self._empty_allocation(outcome_prices, 0, True)

        price_sum = sum(outcome_prices)
        if price_sum >= 1.0:
            # No arb: prices sum to >= $1.00
            return self._empty_allocation(outcome_prices, 0, True)

        # Validate order books if provided
        if order_books is not None:
            if len(order_books) != k:
                raise ValueError(
                    f"Expected {k} order books, got {len(order_books)}"
                )

        # =====================================================
        # Phase 1: Analytical optimal for the price-only model
        # =====================================================
        # To maximize min(shares), we want all shares equal:
        #   shares_i = a_i / p_i = S  (same for all i)
        # So a_i = S * p_i, and sum(a_i) = S * sum(p_i) = B
        # => S = B / sum(p_i)
        # => a_i = B * p_i / sum(p_i)
        allocation = [
            self.budget * p / price_sum for p in outcome_prices
        ]

        # Check if this base allocation is profitable after fees
        base_profit = self._compute_profit(
            allocation, outcome_prices, order_books
        )

        # If not profitable even at the analytical optimum, try scaling down
        if base_profit <= 0 and order_books is None:
            # For the simple model, profit scales linearly with budget.
            # If it's not profitable at full budget, it won't be at any scale.
            # This means fee_rate >= gap / price_sum, so no arb after fees.
            return self._build_result(
                [0.0] * k, outcome_prices, order_books, 0, True
            )

        best_allocation = list(allocation)
        best_profit = base_profit
        converged = True
        iterations_used = 0

        # =====================================================
        # Phase 2: Frank-Wolfe coordinate refinement
        # =====================================================
        # Only needed when order books introduce slippage
        if order_books is not None:
            converged = False

            for iteration in range(1, self.max_iterations + 1):
                iterations_used = iteration

                # Compute gradient: marginal profit from spending $1 more
                # on each outcome
                gradient = self._compute_gradient(
                    allocation, outcome_prices, order_books
                )

                # Find the best and worst outcomes by gradient
                best_idx = 0
                worst_idx = 0
                for i in range(k):
                    if gradient[i] > gradient[best_idx]:
                        best_idx = i
                    if gradient[i] < gradient[worst_idx]:
                        worst_idx = i

                # If all gradients are similar, we've converged
                grad_range = gradient[best_idx] - gradient[worst_idx]
                if grad_range < self.tolerance:
                    converged = True
                    break

                # Away-step Frank-Wolfe: move budget from worst to best
                # Step size: transfer a fraction of the worst outcome's
                # allocation to the best outcome
                step_size = 2.0 / (iteration + 2)
                transfer = allocation[worst_idx] * step_size

                if transfer < self.tolerance:
                    converged = True
                    break

                new_allocation = list(allocation)
                new_allocation[worst_idx] -= transfer
                new_allocation[best_idx] += transfer

                # Bregman projection to maintain feasibility
                new_allocation = self._bregman_project(
                    new_allocation, self.budget
                )

                new_profit = self._compute_profit(
                    new_allocation, outcome_prices, order_books
                )

                if new_profit > best_profit:
                    best_profit = new_profit
                    best_allocation = list(new_allocation)

                # Check convergence
                old_profit = self._compute_profit(
                    allocation, outcome_prices, order_books
                )
                if abs(new_profit - old_profit) < self.tolerance:
                    converged = True
                    allocation = new_allocation
                    break

                allocation = new_allocation

        # =====================================================
        # Phase 3: Budget scale search
        # =====================================================
        # With slippage, using less than the full budget may be better.
        # Also handles the case where FW didn't fully converge.
        allocation = self._search_best_scale(
            best_allocation, outcome_prices, order_books
        )

        # Final profit check
        final_profit = self._compute_profit(
            allocation, outcome_prices, order_books
        )

        # If final profit <= 0, return zero allocation
        if final_profit <= 0:
            return self._build_result(
                [0.0] * k, outcome_prices, order_books,
                iterations_used, converged
            )

        return self._build_result(
            allocation, outcome_prices, order_books,
            iterations_used, converged
        )

    # ------------------------------------------------------------------
    # Core computation helpers
    # ------------------------------------------------------------------

    def _compute_profit(
        self,
        allocation: list[float],
        prices: list[float],
        order_books: Optional[list[OrderBook]],
    ) -> float:
        """Compute guaranteed profit for a given allocation."""
        shares = self._compute_shares(allocation, prices, order_books)
        if not shares:
            return -1e9
        min_shares = min(shares)
        total_cost = sum(allocation)
        gross = min_shares * 1.0 - total_cost
        fees = total_cost * self.fee_rate
        return gross - fees

    def _compute_gradient(
        self,
        allocation: list[float],
        prices: list[float],
        order_books: Optional[list[OrderBook]],
    ) -> list[float]:
        """
        Compute the gradient of profit w.r.t. each allocation component.

        Uses finite differences: perturb each allocation by a small
        epsilon and measure the change in profit. This handles both the
        simple price model and the order-book slippage model uniformly.
        """
        k = len(allocation)
        eps = max(self.budget * 1e-5, 1e-8)
        base_profit = self._compute_profit(allocation, prices, order_books)

        gradient = []
        for i in range(k):
            perturbed = list(allocation)
            perturbed[i] += eps
            # Keep feasible: if over budget, scale down proportionally
            total = sum(perturbed)
            if total > self.budget:
                scale = self.budget / total
                perturbed = [a * scale for a in perturbed]
            new_profit = self._compute_profit(perturbed, prices, order_books)
            gradient.append((new_profit - base_profit) / eps)

        return gradient

    def _compute_shares(
        self,
        allocation: list[float],
        prices: list[float],
        order_books: Optional[list[OrderBook]],
    ) -> list[float]:
        """
        Given dollar allocations, compute shares received per outcome.

        Simple mode (no order books): shares_i = allocation_i / price_i
        With order books: walk the ask side to account for slippage.
        """
        k = len(allocation)
        shares = []

        for i in range(k):
            spend = allocation[i]
            if spend <= 0:
                shares.append(0.0)
                continue

            if order_books is not None and order_books[i].asks:
                # Walk the order book for accurate slippage
                s = self._walk_order_book(order_books[i], spend)
                shares.append(s)
            else:
                # Simple model: infinite liquidity at quoted price
                if prices[i] > 0:
                    shares.append(spend / prices[i])
                else:
                    shares.append(0.0)

        return shares

    def _walk_order_book(self, order_book: OrderBook, spend_usd: float) -> float:
        """
        Simulate buying `spend_usd` worth of tokens by walking through
        the ask side of the order book.

        Returns the number of shares you'd actually receive.

        The ask side is sorted lowest price first (best price first).
        At each level, we buy as many shares as we can afford (or as
        many as are available), then move to the next (worse) price level.
        """
        if spend_usd <= 0:
            return 0.0

        remaining_usd = spend_usd
        total_shares = 0.0

        for level in order_book.asks:
            if remaining_usd <= 0:
                break

            price = level.price
            available_shares = level.size

            if price <= 0 or available_shares <= 0:
                continue

            # Cost to buy all available shares at this level
            cost_for_all = price * available_shares

            if cost_for_all <= remaining_usd:
                # Buy everything at this level
                total_shares += available_shares
                remaining_usd -= cost_for_all
            else:
                # Buy as many shares as we can afford at this price
                shares_we_can_buy = remaining_usd / price
                total_shares += shares_we_can_buy
                remaining_usd = 0.0

        return total_shares

    # ------------------------------------------------------------------
    # Bregman Projection
    # ------------------------------------------------------------------

    def _bregman_project(
        self,
        allocation: list[float],
        budget: float,
    ) -> list[float]:
        """
        Project allocation onto the feasible set using KL-divergence-based
        Bregman projection.

        The feasible set is: a_i >= 0, sum(a_i) <= budget.

        Full KL-divergence projection would solve:
            min_{y in C} sum_i y_i * log(y_i / x_i) - y_i + x_i

        For the budget simplex constraint, this simplifies to proportional
        scaling after clipping negatives (because the KL-optimal projection
        onto a scaled simplex preserves ratios).

        Steps:
        1. Clip any negative values to a small positive floor.
        2. If sum exceeds budget, scale proportionally.
        """
        floor = budget * 1e-10

        projected = [max(a, floor) for a in allocation]

        total = sum(projected)
        if total > budget and total > 0:
            scale = budget / total
            projected = [a * scale for a in projected]

        return projected

    # ------------------------------------------------------------------
    # Budget scale search
    # ------------------------------------------------------------------

    def _search_best_scale(
        self,
        allocation: list[float],
        prices: list[float],
        order_books: Optional[list[OrderBook]],
    ) -> list[float]:
        """
        Search over budget utilization levels to find the optimum.

        With order book slippage, spending 100% of the budget may not
        be optimal because marginal costs increase as you eat through
        the book. We try scale factors from 5% to 100% in 5% increments,
        then refine around the best with 1% increments.

        Also tries rebalancing shares to equalize them (since profit is
        determined by the minimum across all outcomes).
        """
        best_alloc = list(allocation)
        best_profit = self._compute_profit(allocation, prices, order_books)

        # First: try rebalancing shares
        shares = self._compute_shares(allocation, prices, order_books)
        if shares and min(shares) > 0:
            balanced = self._balance_shares(
                allocation, prices, order_books, shares
            )
            balanced = self._bregman_project(balanced, self.budget)
            balanced_profit = self._compute_profit(
                balanced, prices, order_books
            )
            if balanced_profit > best_profit:
                best_profit = balanced_profit
                best_alloc = list(balanced)

        # Coarse search: 5% increments
        best_pct = 100
        for pct in range(5, 105, 5):
            scale = pct / 100.0
            scaled = [a * scale for a in allocation]
            scaled = self._bregman_project(scaled, self.budget)
            profit = self._compute_profit(scaled, prices, order_books)
            if profit > best_profit:
                best_profit = profit
                best_alloc = list(scaled)
                best_pct = pct

        # Fine search: 1% increments around the best
        for pct in range(max(1, best_pct - 5), min(101, best_pct + 6)):
            scale = pct / 100.0
            scaled = [a * scale for a in allocation]
            scaled = self._bregman_project(scaled, self.budget)
            profit = self._compute_profit(scaled, prices, order_books)
            if profit > best_profit:
                best_profit = profit
                best_alloc = list(scaled)

        return best_alloc

    def _balance_shares(
        self,
        allocation: list[float],
        prices: list[float],
        order_books: Optional[list[OrderBook]],
        shares: list[float],
    ) -> list[float]:
        """
        Rebalance allocations to equalize shares across outcomes.

        The profit is determined by min(shares). If outcome A has 100
        shares and outcome B has 50 shares, the extra 50 shares of A
        are wasted. We redirect some of A's budget to B.

        Strategy: iteratively shift budget from outcomes with excess
        shares to those at the minimum.
        """
        k = len(allocation)
        if k == 0:
            return allocation

        balanced = list(allocation)

        for _ in range(20):
            current_shares = self._compute_shares(
                balanced, prices, order_books
            )
            if not current_shares or min(current_shares) <= 0:
                break

            current_min = min(current_shares)
            excess_total = 0.0
            deficit_indices = []

            for i in range(k):
                excess = current_shares[i] - current_min
                if excess > current_min * 0.01:
                    if order_books is not None and order_books[i].asks:
                        avg_price = (
                            balanced[i] / current_shares[i]
                            if current_shares[i] > 0 else prices[i]
                        )
                    else:
                        avg_price = prices[i]
                    save = excess * avg_price * 0.5
                    balanced[i] -= save
                    balanced[i] = max(balanced[i], 0.001)
                    excess_total += save

            if excess_total > 0:
                for i in range(k):
                    if current_shares[i] <= current_min * 1.01:
                        deficit_indices.append(i)

                if deficit_indices:
                    per_deficit = excess_total / len(deficit_indices)
                    for i in deficit_indices:
                        balanced[i] += per_deficit

        return balanced

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(
        self,
        allocation: list[float],
        prices: list[float],
        order_books: Optional[list[OrderBook]],
        iterations: int,
        converged: bool,
    ) -> ArbAllocation:
        """Assemble the final ArbAllocation from an optimized allocation."""
        shares = self._compute_shares(allocation, prices, order_books)
        min_shares = min(shares) if shares else 0.0
        total_cost = sum(allocation)
        gross_profit = min_shares * 1.0 - total_cost
        fees = total_cost * self.fee_rate
        net_profit = gross_profit - fees
        roi = net_profit / total_cost if total_cost > 0 else 0.0

        per_outcome = []
        for i in range(len(prices)):
            per_outcome.append({
                "outcome_index": i,
                "price": prices[i],
                "allocation": allocation[i],
                "shares": shares[i] if i < len(shares) else 0.0,
            })

        return ArbAllocation(
            allocations=list(allocation),
            shares_per_outcome=list(shares),
            min_shares=min_shares,
            total_cost=total_cost,
            gross_profit=gross_profit,
            fees=fees,
            net_profit=net_profit,
            roi=roi,
            iterations=iterations,
            converged=converged,
            per_outcome=per_outcome,
        )

    def _empty_allocation(
        self,
        prices: list[float],
        iterations: int,
        converged: bool,
    ) -> ArbAllocation:
        """Return a zero-allocation result (no arb found or invalid input)."""
        k = len(prices)
        per_outcome = [
            {
                "outcome_index": i,
                "price": prices[i] if i < k else 0.0,
                "allocation": 0.0,
                "shares": 0.0,
            }
            for i in range(k)
        ]
        return ArbAllocation(
            allocations=[0.0] * k,
            shares_per_outcome=[0.0] * k,
            min_shares=0.0,
            total_cost=0.0,
            gross_profit=0.0,
            fees=0.0,
            net_profit=0.0,
            roi=0.0,
            iterations=iterations,
            converged=converged,
            per_outcome=per_outcome,
        )
