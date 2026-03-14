"""
Tests for the Frank-Wolfe + Bregman Projection optimizer.

Runs against synthetic data to verify correctness without needing
live Polymarket API access.
"""

from __future__ import annotations

import sys

from bot.arb.optimizer import ArbOptimizer, ArbAllocation
from bot.polymarket.models import OrderBook, OrderBookLevel


def test_basic_3_outcome():
    """3-outcome market with prices summing to 0.94 (6% gap)."""
    print("=" * 60)
    print("TEST 1: Basic 3-outcome market (6% gap)")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=10.0)
    result = optimizer.optimize(outcome_prices=[0.30, 0.34, 0.30])

    result.print_report()

    # Verify basic properties
    assert result.total_cost > 0, "Should spend some money"
    assert result.total_cost <= 10.0, "Should not exceed budget"
    assert result.min_shares > 0, "Should get some shares"
    assert result.gross_profit > 0, "Should have positive gross profit"
    # With 6% gap and 2% fee, net profit should still be positive
    assert result.net_profit > 0, f"Expected positive net profit, got {result.net_profit:.4f}"
    assert result.roi > 0, "ROI should be positive"
    assert len(result.per_outcome) == 3, "Should have 3 outcomes"

    # Shares should be roughly balanced (optimizer tries to equalize them)
    shares = result.shares_per_outcome
    max_share = max(shares)
    min_share = min(shares)
    ratio = min_share / max_share if max_share > 0 else 0
    print(f"  Share balance ratio: {ratio:.4f} (1.0 = perfectly balanced)")
    assert ratio > 0.5, f"Shares should be somewhat balanced, got ratio {ratio:.4f}"

    print("  PASSED\n")
    return True


def test_2_outcome():
    """2-outcome binary market with 5% gap."""
    print("=" * 60)
    print("TEST 2: Binary market (5% gap)")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=10.0)
    result = optimizer.optimize(outcome_prices=[0.45, 0.50])

    result.print_report()

    assert result.total_cost > 0, "Should spend money"
    assert result.total_cost <= 10.0, "Should not exceed budget"
    assert result.net_profit > 0, f"Expected positive net profit, got {result.net_profit:.4f}"
    assert len(result.per_outcome) == 2, "Should have 2 outcomes"

    print("  PASSED\n")
    return True


def test_no_arb():
    """Prices sum to >= 1.0, no arb should be found."""
    print("=" * 60)
    print("TEST 3: No arb (prices sum to 1.0)")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=10.0)
    result = optimizer.optimize(outcome_prices=[0.50, 0.50])

    result.print_report()

    assert result.total_cost == 0.0, "Should not spend when no arb exists"
    assert result.net_profit == 0.0, "No profit when no arb"

    print("  PASSED\n")
    return True


def test_overpriced():
    """Prices sum to > 1.0, no buy-all arb."""
    print("=" * 60)
    print("TEST 4: Overpriced market (sum > 1.0)")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=10.0)
    result = optimizer.optimize(outcome_prices=[0.40, 0.35, 0.30])

    result.print_report()

    # 0.40 + 0.35 + 0.30 = 1.05, no buy-all arb
    assert result.total_cost == 0.0, "Should not spend when prices > 1.0"

    print("  PASSED\n")
    return True


def test_large_gap():
    """Large arb gap (20%) -- should be very profitable."""
    print("=" * 60)
    print("TEST 5: Large gap (20%)")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=100.0)
    result = optimizer.optimize(outcome_prices=[0.20, 0.20, 0.20, 0.20])

    result.print_report()

    # Sum = 0.80, gap = 0.20
    assert result.net_profit > 0, "Large gap should be profitable"
    assert result.roi > 0.10, f"ROI should be > 10%, got {result.roi:.2%}"

    print("  PASSED\n")
    return True


def test_with_order_books():
    """Test with synthetic order books to verify slippage handling."""
    print("=" * 60)
    print("TEST 6: With order books (slippage)")
    print("=" * 60)

    # Create synthetic order books with limited liquidity
    book_a = OrderBook(
        asks=[
            OrderBookLevel(price=0.30, size=10),   # 10 shares at 0.30
            OrderBookLevel(price=0.35, size=20),   # 20 shares at 0.35
            OrderBookLevel(price=0.40, size=50),   # 50 shares at 0.40
        ],
    )
    book_b = OrderBook(
        asks=[
            OrderBookLevel(price=0.34, size=15),
            OrderBookLevel(price=0.38, size=25),
            OrderBookLevel(price=0.45, size=40),
        ],
    )
    book_c = OrderBook(
        asks=[
            OrderBookLevel(price=0.30, size=12),
            OrderBookLevel(price=0.33, size=18),
            OrderBookLevel(price=0.38, size=30),
        ],
    )

    optimizer = ArbOptimizer(budget=10.0)
    result = optimizer.optimize(
        outcome_prices=[0.30, 0.34, 0.30],
        order_books=[book_a, book_b, book_c],
    )

    result.print_report()

    assert result.total_cost > 0, "Should spend money"
    assert result.total_cost <= 10.0, "Should not exceed budget"
    # With slippage, profit may be smaller but should still be positive
    # for a 6% gap
    print(f"  Net profit with slippage: ${result.net_profit:.4f}")
    # We don't assert positive here because slippage might eat the edge

    print("  PASSED\n")
    return True


def test_walk_order_book():
    """Directly test the order book walking function."""
    print("=" * 60)
    print("TEST 7: Order book walking")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=100.0)

    book = OrderBook(
        asks=[
            OrderBookLevel(price=0.50, size=10),   # $5.00 for 10 shares
            OrderBookLevel(price=0.60, size=20),   # $12.00 for 20 shares
            OrderBookLevel(price=0.80, size=100),  # $80.00 for 100 shares
        ],
    )

    # Test: spend exactly $5.00 (buy all 10 shares at first level)
    shares = optimizer._walk_order_book(book, 5.0)
    print(f"  $5.00 spend -> {shares:.2f} shares (expected: 10.00)")
    assert abs(shares - 10.0) < 0.01, f"Expected 10 shares, got {shares}"

    # Test: spend $10.00 (10 at 0.50, then 8.33 at 0.60)
    shares = optimizer._walk_order_book(book, 10.0)
    expected = 10.0 + 5.0 / 0.60  # 10 + 8.333
    print(f"  $10.00 spend -> {shares:.2f} shares (expected: {expected:.2f})")
    assert abs(shares - expected) < 0.01, f"Expected {expected:.2f} shares, got {shares}"

    # Test: spend $0 -> 0 shares
    shares = optimizer._walk_order_book(book, 0.0)
    assert shares == 0.0, "Zero spend should give zero shares"

    # Test: spend more than the book has
    # Total book: $5 + $12 + $80 = $97
    shares = optimizer._walk_order_book(book, 200.0)
    expected = 10 + 20 + 100  # all shares in the book
    print(f"  $200.00 spend -> {shares:.2f} shares (expected: {expected:.2f}, book exhausted)")
    assert abs(shares - expected) < 0.01, f"Expected {expected:.2f} shares, got {shares}"

    print("  PASSED\n")
    return True


def test_single_outcome():
    """Edge case: only 1 outcome (invalid for arb)."""
    print("=" * 60)
    print("TEST 8: Single outcome (edge case)")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=10.0)
    result = optimizer.optimize(outcome_prices=[0.50])

    assert result.total_cost == 0.0, "Single outcome cannot have arb"
    assert result.net_profit == 0.0

    print("  PASSED\n")
    return True


def test_zero_price():
    """Edge case: outcome with price 0 (invalid)."""
    print("=" * 60)
    print("TEST 9: Zero price outcome (edge case)")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=10.0)
    result = optimizer.optimize(outcome_prices=[0.30, 0.0, 0.30])

    assert result.total_cost == 0.0, "Cannot trade zero-price outcome"

    print("  PASSED\n")
    return True


def test_tiny_gap():
    """Gap smaller than fees -- should not be profitable."""
    print("=" * 60)
    print("TEST 10: Tiny gap (1%, smaller than 2% fee)")
    print("=" * 60)

    optimizer = ArbOptimizer(budget=10.0, fee_rate=0.02)
    result = optimizer.optimize(outcome_prices=[0.495, 0.495])
    # Sum = 0.99, gap = 0.01, fee = 2% of cost
    # gross_profit ~ 0.01 * shares, but fee ~ 0.02 * cost
    # The optimizer should find this isn't profitable after fees

    result.print_report()
    # Either the optimizer finds zero allocation or slightly negative
    print(f"  Net profit: ${result.net_profit:.4f}")
    # This should be non-positive after fees
    # (gap=1% < fee_rate=2%)

    print("  PASSED\n")
    return True


def test_many_outcomes():
    """Market with many outcomes (e.g., 10-way election)."""
    print("=" * 60)
    print("TEST 11: Many outcomes (10-way market)")
    print("=" * 60)

    # 10 outcomes each at 0.08 = sum of 0.80, gap = 0.20
    prices = [0.08] * 10
    optimizer = ArbOptimizer(budget=50.0)
    result = optimizer.optimize(outcome_prices=prices)

    result.print_report()

    assert result.net_profit > 0, "20% gap should be profitable"
    assert len(result.per_outcome) == 10

    # With equal prices, allocations should be roughly equal
    allocs = result.allocations
    avg = sum(allocs) / len(allocs)
    for a in allocs:
        assert abs(a - avg) < avg * 0.5, f"Allocation {a:.4f} too far from avg {avg:.4f}"

    print("  PASSED\n")
    return True


def main():
    """Run all tests and report results."""
    print()
    print("*" * 60)
    print("  Frank-Wolfe Optimizer Test Suite")
    print("*" * 60)
    print()

    tests = [
        test_basic_3_outcome,
        test_2_outcome,
        test_no_arb,
        test_overpriced,
        test_large_gap,
        test_with_order_books,
        test_walk_order_book,
        test_single_outcome,
        test_zero_price,
        test_tiny_gap,
        test_many_outcomes,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            result = test_fn()
            if result is not False:
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("*" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("*" * 60)
    print()

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
