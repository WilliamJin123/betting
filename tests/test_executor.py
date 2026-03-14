"""
Tests for the OrderExecutor (paper trading mode).

These tests verify that:
1. Paper orders are placed and tracked correctly
2. Balance is updated on buy/sell
3. Insufficient balance is rejected
4. Order cancellation works
5. Order status lookup works
6. Input validation catches bad parameters
7. Auth module loads/validates correctly
"""

import os
import sys

# Ensure the project root is on sys.path so `bot` can be imported.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bot.polymarket.executor import OrderExecutor
from bot.polymarket.auth import (
    PolymarketCredentials,
    validate_credentials,
    load_credentials,
    ENV_PRIVATE_KEY,
)


# ======================================================================
# OrderExecutor — Paper Mode Tests
# ======================================================================

def test_paper_mode_init():
    """Executor initializes in paper mode with correct default balance."""
    ex = OrderExecutor(mode="paper", initial_bankroll=10.0)
    assert ex.mode == "paper"
    assert ex.get_balance() == 10.0
    print("  PASS: test_paper_mode_init")


def test_paper_place_buy_order():
    """Placing a BUY order reduces the balance and records the order."""
    ex = OrderExecutor(mode="paper", initial_bankroll=10.0)

    result = ex.place_order(
        token_id="token123",
        side="BUY",
        size_usd=2.50,
        price=0.50,
        market_question="Test market?",
    )

    assert result["status"] == "FILLED"
    assert result["order_id"] is not None
    assert result["order_id"].startswith("paper-")
    assert result["side"] == "BUY"
    assert result["size_usd"] == 2.50
    assert result["price"] == 0.50
    assert result["shares"] == 5.0  # 2.50 / 0.50
    assert result["mode"] == "paper"

    # Balance should be reduced
    assert ex.get_balance() == 7.50

    # Position should be tracked
    positions = ex.get_positions()
    assert len(positions) == 1
    assert positions[0]["token_id"] == "token123"
    assert positions[0]["shares"] == 5.0

    print("  PASS: test_paper_place_buy_order")


def test_paper_place_sell_order():
    """Placing a SELL order increases the balance."""
    ex = OrderExecutor(mode="paper", initial_bankroll=10.0)

    # First buy, then sell
    ex.place_order("token123", "BUY", 3.0, 0.60)
    assert ex.get_balance() == 7.0

    result = ex.place_order("token123", "SELL", 3.0, 0.70)
    assert result["status"] == "FILLED"
    assert ex.get_balance() == 10.0  # 7.0 + 3.0

    print("  PASS: test_paper_place_sell_order")


def test_paper_insufficient_balance():
    """Trying to buy more than the balance should be rejected."""
    ex = OrderExecutor(mode="paper", initial_bankroll=5.0)

    result = ex.place_order("token123", "BUY", 10.0, 0.50)
    assert result["status"] == "REJECTED"
    assert result["reason"] == "insufficient_balance"
    assert result["order_id"] is None

    # Balance should be unchanged
    assert ex.get_balance() == 5.0

    print("  PASS: test_paper_insufficient_balance")


def test_paper_multiple_orders():
    """Multiple orders should correctly update the balance."""
    ex = OrderExecutor(mode="paper", initial_bankroll=10.0)

    ex.place_order("token_a", "BUY", 2.0, 0.40, "Market A")
    ex.place_order("token_b", "BUY", 3.0, 0.60, "Market B")
    ex.place_order("token_c", "BUY", 1.0, 0.80, "Market C")

    assert abs(ex.get_balance() - 4.0) < 0.001  # 10 - 2 - 3 - 1

    positions = ex.get_positions()
    assert len(positions) == 3

    history = ex.get_order_history()
    assert len(history) == 3

    print("  PASS: test_paper_multiple_orders")


def test_paper_order_status():
    """get_order_status returns correct info for placed orders."""
    ex = OrderExecutor(mode="paper", initial_bankroll=10.0)

    result = ex.place_order("token123", "BUY", 1.0, 0.50)
    order_id = result["order_id"]

    status = ex.get_order_status(order_id)
    assert status["status"] == "FILLED"
    assert status["order_id"] == order_id
    assert status["mode"] == "paper"

    # Non-existent order
    status2 = ex.get_order_status("fake-order-id")
    assert status2["status"] == "NOT_FOUND"

    print("  PASS: test_paper_order_status")


def test_paper_cancel_order():
    """
    Paper orders are filled immediately, so cancel should fail.
    This tests the cancel logic path.
    """
    ex = OrderExecutor(mode="paper", initial_bankroll=10.0)

    result = ex.place_order("token123", "BUY", 1.0, 0.50)
    order_id = result["order_id"]

    # Can't cancel a filled order
    cancelled = ex.cancel_order(order_id)
    assert cancelled is False

    # Can't cancel a non-existent order
    cancelled2 = ex.cancel_order("fake-order")
    assert cancelled2 is False

    print("  PASS: test_paper_cancel_order")


def test_paper_reset():
    """reset_paper() should clear all state."""
    ex = OrderExecutor(mode="paper", initial_bankroll=10.0)

    ex.place_order("token123", "BUY", 5.0, 0.50)
    assert ex.get_balance() == 5.0
    assert len(ex.get_positions()) == 1

    ex.reset_paper()

    assert ex.get_balance() == 10.0
    assert len(ex.get_positions()) == 0
    assert len(ex.get_order_history()) == 0

    print("  PASS: test_paper_reset")


# ======================================================================
# Input Validation Tests
# ======================================================================

def test_invalid_mode():
    """Invalid mode should raise ValueError."""
    try:
        OrderExecutor(mode="yolo")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid mode" in str(e)
    print("  PASS: test_invalid_mode")


def test_invalid_side():
    """Invalid side should raise ValueError."""
    ex = OrderExecutor(mode="paper")
    try:
        ex.place_order("token123", "HOLD", 1.0, 0.50)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid side" in str(e)
    print("  PASS: test_invalid_side")


def test_invalid_price():
    """Price outside 0.01-0.99 should raise ValueError."""
    ex = OrderExecutor(mode="paper")

    for bad_price in [0.0, -0.5, 1.0, 1.5, 0.001]:
        try:
            ex.place_order("token123", "BUY", 1.0, bad_price)
            assert False, f"Should have raised ValueError for price={bad_price}"
        except ValueError as e:
            assert "price" in str(e).lower()

    print("  PASS: test_invalid_price")


def test_invalid_size():
    """Non-positive size should raise ValueError."""
    ex = OrderExecutor(mode="paper")

    for bad_size in [0, -1.0, -100]:
        try:
            ex.place_order("token123", "BUY", bad_size, 0.50)
            assert False, f"Should have raised ValueError for size={bad_size}"
        except ValueError as e:
            assert "size_usd" in str(e)

    print("  PASS: test_invalid_size")


# ======================================================================
# Auth Module Tests
# ======================================================================

def test_credentials_dataclass():
    """PolymarketCredentials should track whether derived creds are set."""
    # No derived creds
    creds = PolymarketCredentials(private_key="0x" + "a" * 64)
    assert not creds.has_derived_creds

    # With derived creds
    creds2 = PolymarketCredentials(
        private_key="0x" + "a" * 64,
        api_key="key",
        api_secret="secret",
        api_passphrase="pass",
    )
    assert creds2.has_derived_creds

    print("  PASS: test_credentials_dataclass")


def test_validate_credentials():
    """validate_credentials should return warnings for issues."""
    # Good creds
    good = PolymarketCredentials(
        private_key="0x" + "a" * 64,
        api_key="key",
        api_secret="secret",
        api_passphrase="pass",
    )
    assert len(validate_credentials(good)) == 0

    # Short key
    short = PolymarketCredentials(private_key="0xabc")
    warnings = validate_credentials(short)
    assert len(warnings) >= 1
    assert any("shorter" in w for w in warnings)

    # Empty key
    empty = PolymarketCredentials(private_key="")
    warnings2 = validate_credentials(empty)
    assert len(warnings2) >= 1
    assert any("empty" in w.lower() for w in warnings2)

    print("  PASS: test_validate_credentials")


def test_load_credentials_missing():
    """load_credentials should raise when POLYMARKET_PRIVATE_KEY is not set."""
    # Remove the env var if it exists
    old_val = os.environ.pop(ENV_PRIVATE_KEY, None)
    try:
        load_credentials()
        assert False, "Should have raised EnvironmentError"
    except EnvironmentError as e:
        assert "POLYMARKET_PRIVATE_KEY" in str(e)
    finally:
        # Restore
        if old_val is not None:
            os.environ[ENV_PRIVATE_KEY] = old_val
    print("  PASS: test_load_credentials_missing")


def test_load_credentials_success():
    """load_credentials should work when private key is set."""
    test_key = "0x" + "ab" * 32  # 64 hex chars
    old_val = os.environ.get(ENV_PRIVATE_KEY)
    os.environ[ENV_PRIVATE_KEY] = test_key
    try:
        creds = load_credentials()
        assert creds.private_key == test_key
        assert not creds.has_derived_creds  # No API creds set
    finally:
        if old_val is not None:
            os.environ[ENV_PRIVATE_KEY] = old_val
        else:
            os.environ.pop(ENV_PRIVATE_KEY, None)
    print("  PASS: test_load_credentials_success")


def test_load_credentials_no_0x_prefix():
    """load_credentials should add 0x prefix if missing."""
    raw_key = "ab" * 32
    old_val = os.environ.get(ENV_PRIVATE_KEY)
    os.environ[ENV_PRIVATE_KEY] = raw_key
    try:
        creds = load_credentials()
        assert creds.private_key == "0x" + raw_key
    finally:
        if old_val is not None:
            os.environ[ENV_PRIVATE_KEY] = old_val
        else:
            os.environ.pop(ENV_PRIVATE_KEY, None)
    print("  PASS: test_load_credentials_no_0x_prefix")


# ======================================================================
# Run all tests
# ======================================================================

def main():
    print("\n" + "=" * 50)
    print("  EXECUTOR & AUTH TESTS")
    print("=" * 50 + "\n")

    tests = [
        # Executor paper mode
        test_paper_mode_init,
        test_paper_place_buy_order,
        test_paper_place_sell_order,
        test_paper_insufficient_balance,
        test_paper_multiple_orders,
        test_paper_order_status,
        test_paper_cancel_order,
        test_paper_reset,
        # Input validation
        test_invalid_mode,
        test_invalid_side,
        test_invalid_price,
        test_invalid_size,
        # Auth module
        test_credentials_dataclass,
        test_validate_credentials,
        test_load_credentials_missing,
        test_load_credentials_success,
        test_load_credentials_no_0x_prefix,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 50}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
