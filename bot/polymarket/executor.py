"""
Order Executor — Places bets on Polymarket (paper or live).
============================================================

This is the bridge between "the bot decided to bet" and "money moves."
It has two modes:

    1. **Paper trading** (default):
       Simulates order placement without touching real money. Maintains a
       virtual bankroll and position tracker. Use this to verify bot logic
       before risking a single dollar.

    2. **Live trading**:
       Uses Polymarket's py-clob-client SDK to place real limit orders on
       the CLOB (Central Limit Order Book). Requires API credentials set
       up via environment variables (see auth.py).

WHY LIMIT ORDERS, NOT MARKET ORDERS:
    Market orders execute immediately at whatever price is available. With
    a $10 bankroll, we can't afford slippage. Limit orders let us specify
    the exact price we're willing to pay. If the price moves away from us,
    the order simply doesn't fill -- which is better than overpaying.

    We use GTC (Good-Til-Cancelled) limit orders: they sit on the book
    until filled or explicitly cancelled.

USAGE:
    # Paper trading (safe, no money at risk):
    from bot.polymarket.executor import OrderExecutor

    executor = OrderExecutor(mode="paper")
    result = executor.place_order(
        token_id="12345...",
        side="BUY",
        size_usd=1.50,
        price=0.60,
        market_question="Will BTC hit $100k?"
    )
    print(result)  # {'order_id': 'paper-...', 'status': 'FILLED', ...}
    print(executor.get_balance())  # 8.50 (started at 10, spent 1.50)

    # Live trading (real money!):
    executor = OrderExecutor(mode="live")
    result = executor.place_order(...)  # Actually submits to Polymarket

PAPER TRADING MECHANICS:
    Paper mode simulates a simplified exchange:
    - Orders are "filled" immediately at the requested price (no partial fills).
    - The simulated bankroll decreases by size_usd when buying.
    - Positions are tracked so you can see open exposure.
    - This is intentionally optimistic -- real fills may be partial or fail.
      The point is to test decision logic, not execution quality.

LIVE TRADING REQUIREMENTS:
    - py-clob-client package: pip install py-clob-client
    - Environment variables set (see auth.py for details):
        POLYMARKET_PRIVATE_KEY (required)
        POLYMARKET_API_KEY, POLYMARKET_API_SECRET, POLYMARKET_API_PASSPHRASE (optional)
    - USDC.e funded wallet on Polygon
    - Token allowances approved (EOA wallets only -- see auth.py docs)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from bot import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper-trade position tracking
# ---------------------------------------------------------------------------

@dataclass
class PaperPosition:
    """
    A simulated position held in paper trading mode.

    Fields:
        token_id:       Which outcome token we hold.
        side:           "BUY" or "SELL".
        size_usd:       How much USD we spent.
        price:          Price per share we paid.
        shares:         Number of shares we hold (size_usd / price).
        market_question: Human-readable label for logging.
        timestamp:      When the position was opened.
    """
    token_id: str
    side: str
    size_usd: float
    price: float
    shares: float
    market_question: str = ""
    timestamp: str = ""


@dataclass
class PaperOrder:
    """
    A simulated order in paper trading mode.

    In paper mode, orders fill immediately, so status is always "FILLED"
    or "CANCELLED". In a more realistic simulator you could add partial
    fills and time-in-force logic, but that's overkill for testing bet
    selection.

    Fields:
        order_id:        Unique identifier (prefixed with "paper-").
        token_id:        The outcome token being traded.
        side:            "BUY" or "SELL".
        size_usd:        Dollar amount of the order.
        price:           Limit price (0.00 to 1.00).
        shares:          Number of shares (size_usd / price).
        status:          "FILLED", "CANCELLED", or "OPEN".
        market_question: Human-readable label.
        timestamp:       When the order was placed.
    """
    order_id: str
    token_id: str
    side: str
    size_usd: float
    price: float
    shares: float
    status: str = "FILLED"
    market_question: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Order Executor
# ---------------------------------------------------------------------------

class OrderExecutor:
    """
    Places orders on Polymarket in either paper or live mode.

    Paper mode (default) simulates everything locally. Live mode uses
    py-clob-client to interact with the real Polymarket CLOB API.

    Args:
        mode:              "paper" (default) or "live".
        initial_bankroll:  Starting balance for paper trading.
                           Defaults to config.INITIAL_BANKROLL ($10).
    """

    VALID_MODES = ("paper", "live")
    VALID_SIDES = ("BUY", "SELL")

    # Polymarket constants
    CLOB_HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137  # Polygon mainnet

    def __init__(
        self,
        mode: str = "paper",
        initial_bankroll: float | None = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Must be one of {self.VALID_MODES}."
            )

        self.mode = mode
        self._initial_bankroll = (
            initial_bankroll if initial_bankroll is not None
            else config.INITIAL_BANKROLL
        )

        # Paper trading state
        self._paper_balance: float = self._initial_bankroll
        self._paper_orders: dict[str, PaperOrder] = {}
        self._paper_positions: list[PaperPosition] = []

        # Live trading client (initialized lazily)
        self._clob_client: Any = None

        if mode == "live":
            self._init_live_client()
        else:
            logger.info(
                f"OrderExecutor started in PAPER mode "
                f"(balance: ${self._paper_balance:.2f})"
            )

    # ======================================================================
    # Public API
    # ======================================================================

    def place_order(
        self,
        token_id: str,
        side: str,
        size_usd: float,
        price: float,
        market_question: str = "",
    ) -> dict[str, Any]:
        """
        Place a bet on Polymarket.

        In paper mode, this simulates the order locally and updates the
        virtual bankroll. In live mode, this creates and submits a signed
        limit order to Polymarket's CLOB.

        Args:
            token_id:        The outcome token to trade. This is the long
                             numeric string from market.yes_token_id or
                             market.no_token_id.
            side:            "BUY" or "SELL".
                             BUY = you're buying shares of this outcome.
                             SELL = you're selling shares you already hold.
            size_usd:        Dollar amount to spend (for BUY) or dollar value
                             of shares to sell (for SELL).
            price:           Limit price between 0.01 and 0.99.
                             For a BUY, this is the max you'll pay per share.
                             For a SELL, this is the min you'll accept.
            market_question: Human-readable market name (for logging only).

        Returns:
            Dict with order details:
                {
                    "order_id": str,     # Unique order identifier
                    "token_id": str,     # Token that was traded
                    "side": str,         # "BUY" or "SELL"
                    "size_usd": float,   # Dollar amount
                    "price": float,      # Limit price
                    "shares": float,     # Number of shares (size_usd / price)
                    "status": str,       # "FILLED", "OPEN", "ERROR", etc.
                    "mode": str,         # "paper" or "live"
                    "market_question": str,
                    "timestamp": str,    # ISO 8601
                }

        Raises:
            ValueError: If parameters are invalid (bad side, price out of
                        range, insufficient balance in paper mode, etc.).
        """
        # ----- Validate inputs -----
        self._validate_order_params(side, size_usd, price)

        if self.mode == "paper":
            return self._paper_place_order(
                token_id, side, size_usd, price, market_question
            )
        else:
            return self._live_place_order(
                token_id, side, size_usd, price, market_question
            )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: The order ID returned by place_order().

        Returns:
            True if the order was successfully cancelled, False if it
            couldn't be cancelled (already filled, not found, etc.).
        """
        if self.mode == "paper":
            return self._paper_cancel_order(order_id)
        else:
            return self._live_cancel_order(order_id)

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Check the current status of an order.

        Args:
            order_id: The order ID returned by place_order().

        Returns:
            Dict with order details including "status" field.
            Status values: "FILLED", "OPEN", "CANCELLED", "NOT_FOUND".
        """
        if self.mode == "paper":
            return self._paper_get_order_status(order_id)
        else:
            return self._live_get_order_status(order_id)

    def get_balance(self) -> float:
        """
        Get the current USDC balance.

        In paper mode, returns the simulated balance (initial bankroll
        minus money spent on open/filled orders).
        In live mode, queries the wallet's actual USDC.e balance.

        Returns:
            Balance in USD as a float.
        """
        if self.mode == "paper":
            return self._paper_balance
        else:
            return self._live_get_balance()

    def get_positions(self) -> list[dict[str, Any]]:
        """
        Get all open positions.

        Returns:
            List of dicts, each describing a position:
                {
                    "token_id": str,
                    "side": str,
                    "size_usd": float,
                    "price": float,
                    "shares": float,
                    "market_question": str,
                    "timestamp": str,
                }
        """
        if self.mode == "paper":
            return [
                {
                    "token_id": p.token_id,
                    "side": p.side,
                    "size_usd": p.size_usd,
                    "price": p.price,
                    "shares": round(p.shares, 4),
                    "market_question": p.market_question,
                    "timestamp": p.timestamp,
                }
                for p in self._paper_positions
            ]
        else:
            return self._live_get_positions()

    def get_order_history(self) -> list[dict[str, Any]]:
        """
        Get all orders placed in this session (paper mode only).

        Returns:
            List of order dicts, most recent first.
        """
        if self.mode == "paper":
            orders = sorted(
                self._paper_orders.values(),
                key=lambda o: o.timestamp,
                reverse=True,
            )
            return [
                {
                    "order_id": o.order_id,
                    "token_id": o.token_id,
                    "side": o.side,
                    "size_usd": o.size_usd,
                    "price": o.price,
                    "shares": round(o.shares, 4),
                    "status": o.status,
                    "market_question": o.market_question,
                    "timestamp": o.timestamp,
                }
                for o in orders
            ]
        else:
            logger.warning("get_order_history() is only available in paper mode.")
            return []

    # ======================================================================
    # Input validation
    # ======================================================================

    def _validate_order_params(
        self, side: str, size_usd: float, price: float
    ) -> None:
        """Validate order parameters before placing."""
        if side not in self.VALID_SIDES:
            raise ValueError(
                f"Invalid side {side!r}. Must be one of {self.VALID_SIDES}."
            )

        if size_usd <= 0:
            raise ValueError(
                f"size_usd must be positive, got {size_usd}."
            )

        if not (0.001 <= price <= 0.999):
            raise ValueError(
                f"price must be between 0.001 and 0.999, got {price}. "
                f"Polymarket prices represent probabilities."
            )

    # ======================================================================
    # Paper Trading Implementation
    # ======================================================================

    def _paper_place_order(
        self,
        token_id: str,
        side: str,
        size_usd: float,
        price: float,
        market_question: str,
    ) -> dict[str, Any]:
        """Simulate placing an order in paper mode."""
        now = datetime.now(timezone.utc).isoformat()
        shares = size_usd / price

        # Check balance for BUY orders
        if side == "BUY":
            if size_usd > self._paper_balance:
                logger.warning(
                    f"PAPER: Insufficient balance. "
                    f"Need ${size_usd:.2f}, have ${self._paper_balance:.2f}"
                )
                return {
                    "order_id": None,
                    "token_id": token_id,
                    "side": side,
                    "size_usd": size_usd,
                    "price": price,
                    "shares": round(shares, 4),
                    "status": "REJECTED",
                    "reason": "insufficient_balance",
                    "mode": "paper",
                    "market_question": market_question,
                    "timestamp": now,
                }

        # Generate a unique order ID
        order_id = f"paper-{uuid.uuid4().hex[:12]}"

        # Create the order (immediately filled in paper mode)
        order = PaperOrder(
            order_id=order_id,
            token_id=token_id,
            side=side,
            size_usd=size_usd,
            price=price,
            shares=shares,
            status="FILLED",
            market_question=market_question,
            timestamp=now,
        )
        self._paper_orders[order_id] = order

        # Update balance and positions for BUY orders
        if side == "BUY":
            self._paper_balance -= size_usd
            self._paper_positions.append(PaperPosition(
                token_id=token_id,
                side=side,
                size_usd=size_usd,
                price=price,
                shares=shares,
                market_question=market_question,
                timestamp=now,
            ))
        elif side == "SELL":
            # For SELL, add the proceeds back to balance
            # (simplified: assumes we have the shares to sell)
            self._paper_balance += size_usd
            # Remove the position if it exists
            self._paper_positions = [
                p for p in self._paper_positions
                if not (p.token_id == token_id and p.side == "BUY")
            ]

        # Log what happened
        logger.info(
            f"PAPER ORDER: {side} {shares:.2f} shares of "
            f"'{market_question or token_id[:20]}' "
            f"@ ${price:.2f} (${size_usd:.2f} total) "
            f"| Balance: ${self._paper_balance:.2f}"
        )

        return {
            "order_id": order_id,
            "token_id": token_id,
            "side": side,
            "size_usd": round(size_usd, 4),
            "price": price,
            "shares": round(shares, 4),
            "status": "FILLED",
            "mode": "paper",
            "market_question": market_question,
            "timestamp": now,
        }

    def _paper_cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order (only possible if not already filled)."""
        order = self._paper_orders.get(order_id)
        if order is None:
            logger.warning(f"PAPER: Order {order_id} not found.")
            return False

        if order.status == "FILLED":
            logger.warning(
                f"PAPER: Cannot cancel order {order_id} -- already filled."
            )
            return False

        if order.status == "CANCELLED":
            logger.warning(
                f"PAPER: Order {order_id} is already cancelled."
            )
            return False

        order.status = "CANCELLED"

        # Refund if it was a BUY order
        if order.side == "BUY":
            self._paper_balance += order.size_usd
            logger.info(
                f"PAPER: Cancelled order {order_id}, "
                f"refunded ${order.size_usd:.2f}. "
                f"Balance: ${self._paper_balance:.2f}"
            )

        return True

    def _paper_get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get status of a paper order."""
        order = self._paper_orders.get(order_id)
        if order is None:
            return {
                "order_id": order_id,
                "status": "NOT_FOUND",
                "mode": "paper",
            }

        return {
            "order_id": order.order_id,
            "token_id": order.token_id,
            "side": order.side,
            "size_usd": order.size_usd,
            "price": order.price,
            "shares": round(order.shares, 4),
            "status": order.status,
            "mode": "paper",
            "market_question": order.market_question,
            "timestamp": order.timestamp,
        }

    # ======================================================================
    # Live Trading Implementation
    # ======================================================================

    def _init_live_client(self) -> None:
        """
        Initialize the py-clob-client for live trading.

        This requires:
        1. py-clob-client to be installed (pip install py-clob-client)
        2. POLYMARKET_PRIVATE_KEY environment variable to be set
        """
        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            raise ImportError(
                "Live trading requires the py-clob-client package.\n"
                "Install it with: pip install py-clob-client\n"
                "Then set your credentials (see auth.py for details)."
            )

        from bot.polymarket.auth import load_credentials

        creds = load_credentials()

        # Build the CLOB client
        # signature_type=0 means standard EOA wallet (MetaMask, etc.)
        self._clob_client = ClobClient(
            self.CLOB_HOST,
            key=creds.private_key,
            chain_id=self.CHAIN_ID,
            signature_type=0,
            funder=creds.funder,
        )

        # Set up API credentials
        if creds.has_derived_creds:
            # Use pre-derived credentials from environment
            from py_clob_client.client import ApiCreds
            api_creds = ApiCreds(
                api_key=creds.api_key,
                api_secret=creds.api_secret,
                api_passphrase=creds.api_passphrase,
            )
            self._clob_client.set_api_creds(api_creds)
            logger.info("Live client initialized with pre-derived API creds.")
        else:
            # Derive credentials from private key (network call)
            logger.info("Deriving API credentials from private key...")
            derived_creds = self._clob_client.create_or_derive_api_creds()
            self._clob_client.set_api_creds(derived_creds)
            logger.info("Live client initialized with freshly derived API creds.")

        # Verify connectivity
        try:
            ok = self._clob_client.get_ok()
            logger.info(f"CLOB API health check: {ok}")
        except Exception as e:
            logger.warning(f"CLOB API health check failed: {e}")

        logger.info("OrderExecutor started in LIVE mode. Real money is at risk!")

    def _live_place_order(
        self,
        token_id: str,
        side: str,
        size_usd: float,
        price: float,
        market_question: str,
    ) -> dict[str, Any]:
        """
        Place a real limit order on Polymarket via py-clob-client.

        Uses create_and_post_order() which:
        1. Constructs the order with the given parameters
        2. Cryptographically signs it with your private key
        3. Submits it to the Polymarket CLOB
        """
        now = datetime.now(timezone.utc).isoformat()
        shares = size_usd / price

        if self._clob_client is None:
            return {
                "order_id": None,
                "status": "ERROR",
                "reason": "Live client not initialized",
                "mode": "live",
                "timestamp": now,
            }

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            # Map our side strings to py-clob-client constants
            clob_side = BUY if side == "BUY" else SELL

            # We need the market's tick_size and neg_risk for proper order
            # construction. Fetch from the CLOB API.
            # The token_id is for a specific outcome; we need the market info.
            # py-clob-client's create_and_post_order handles this via options.
            #
            # For now, we fetch the market info to get tick_size and neg_risk.
            # This adds a network call, but it's necessary for correct pricing.
            tick_size = "0.01"  # Default
            neg_risk = False
            try:
                # get_market expects a condition_id, but we have a token_id.
                # We'll use default tick_size and catch errors.
                # In production, you'd look this up from the market data
                # you already fetched in the scanning phase.
                pass  # TODO: Pass tick_size and neg_risk from caller's market data
            except Exception:
                logger.warning(
                    "Could not fetch market info for tick_size/neg_risk. "
                    "Using defaults."
                )

            logger.info(
                f"LIVE ORDER: Submitting {side} {shares:.2f} shares "
                f"@ ${price:.2f} (${size_usd:.2f}) "
                f"for '{market_question or token_id[:20]}'"
            )

            # Place the order
            # size in py-clob-client is share count, not USD
            response = self._clob_client.create_and_post_order(
                OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=shares,
                    side=clob_side,
                    order_type=OrderType.GTC,
                ),
                options={
                    "tick_size": tick_size,
                    "neg_risk": neg_risk,
                },
            )

            order_id = response.get("orderID", response.get("id", "unknown"))
            status = response.get("status", "UNKNOWN")

            logger.info(
                f"LIVE ORDER RESPONSE: order_id={order_id}, status={status}"
            )

            return {
                "order_id": order_id,
                "token_id": token_id,
                "side": side,
                "size_usd": round(size_usd, 4),
                "price": price,
                "shares": round(shares, 4),
                "status": status,
                "mode": "live",
                "market_question": market_question,
                "timestamp": now,
                "raw_response": response,
            }

        except Exception as e:
            logger.error(f"LIVE ORDER FAILED: {e}", exc_info=True)
            return {
                "order_id": None,
                "token_id": token_id,
                "side": side,
                "size_usd": size_usd,
                "price": price,
                "shares": round(shares, 4),
                "status": "ERROR",
                "reason": str(e),
                "mode": "live",
                "market_question": market_question,
                "timestamp": now,
            }

    def _live_cancel_order(self, order_id: str) -> bool:
        """Cancel a live order on Polymarket."""
        if self._clob_client is None:
            logger.error("Live client not initialized.")
            return False

        try:
            self._clob_client.cancel(order_id)
            logger.info(f"LIVE: Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"LIVE: Failed to cancel order {order_id}: {e}")
            return False

    def _live_get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get the status of a live order from Polymarket."""
        if self._clob_client is None:
            return {
                "order_id": order_id,
                "status": "ERROR",
                "reason": "Live client not initialized",
                "mode": "live",
            }

        try:
            # py-clob-client uses get_order to fetch a specific order
            order = self._clob_client.get_order(order_id)
            return {
                "order_id": order_id,
                "status": order.get("status", "UNKNOWN"),
                "mode": "live",
                "raw_response": order,
            }
        except Exception as e:
            logger.error(f"LIVE: Failed to get order status: {e}")
            return {
                "order_id": order_id,
                "status": "ERROR",
                "reason": str(e),
                "mode": "live",
            }

    def _live_get_balance(self) -> float:
        """
        Get the wallet's USDC.e balance on Polygon.

        NOTE: py-clob-client does not have a built-in balance-check method.
        Checking on-chain balances requires a web3 call to the USDC.e contract
        on Polygon. For now, this returns -1.0 as a sentinel value indicating
        "not implemented" and logs a warning.

        TODO: Implement using web3.py to query the USDC.e token contract:
            USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            contract = w3.eth.contract(address=USDC_E_ADDRESS, abi=ERC20_ABI)
            balance = contract.functions.balanceOf(wallet_address).call()
            return balance / 1e6  # USDC.e has 6 decimals
        """
        logger.warning(
            "get_balance() in live mode is not yet implemented. "
            "Check your wallet balance manually or via Polygonscan."
        )
        return -1.0

    def _live_get_positions(self) -> list[dict[str, Any]]:
        """
        Get open positions from Polymarket.

        TODO: Implement using Polymarket's Data API or on-chain queries.
        The Data API at https://data-api.polymarket.com has position endpoints
        but requires knowing the user's wallet address.
        """
        logger.warning(
            "get_positions() in live mode is not yet implemented."
        )
        return []

    # ======================================================================
    # Utility methods
    # ======================================================================

    def reset_paper(self) -> None:
        """
        Reset paper trading state to initial values.

        Clears all orders, positions, and resets the balance to the
        initial bankroll. Useful for starting a fresh simulation.
        """
        if self.mode != "paper":
            logger.warning("reset_paper() only works in paper mode.")
            return

        self._paper_balance = self._initial_bankroll
        self._paper_orders.clear()
        self._paper_positions.clear()
        logger.info(
            f"Paper trading state reset. Balance: ${self._paper_balance:.2f}"
        )

    def print_status(self) -> None:
        """Print a human-readable summary of the executor's current state."""
        print(f"\n{'=' * 50}")
        print(f"  ORDER EXECUTOR STATUS ({self.mode.upper()} MODE)")
        print(f"{'=' * 50}")

        if self.mode == "paper":
            print(f"\n  Starting balance:  ${self._initial_bankroll:.2f}")
            print(f"  Current balance:   ${self._paper_balance:.2f}")
            spent = self._initial_bankroll - self._paper_balance
            print(f"  Capital deployed:  ${spent:.2f}")

            filled = sum(
                1 for o in self._paper_orders.values() if o.status == "FILLED"
            )
            cancelled = sum(
                1 for o in self._paper_orders.values() if o.status == "CANCELLED"
            )
            print(f"\n  Orders placed:     {len(self._paper_orders)}")
            print(f"  Orders filled:     {filled}")
            print(f"  Orders cancelled:  {cancelled}")
            print(f"  Open positions:    {len(self._paper_positions)}")

            if self._paper_positions:
                print(f"\n  Positions:")
                for p in self._paper_positions:
                    label = p.market_question or p.token_id[:20]
                    print(
                        f"    {p.side} {p.shares:.2f} shares "
                        f"@ ${p.price:.2f} "
                        f"(${p.size_usd:.2f}) — {label}"
                    )
        else:
            print("\n  Live mode — check your wallet for balance/positions.")
            balance = self.get_balance()
            if balance >= 0:
                print(f"  USDC balance: ${balance:.2f}")

        print(f"{'=' * 50}\n")

    def __repr__(self) -> str:
        if self.mode == "paper":
            return (
                f"OrderExecutor(mode='paper', "
                f"balance=${self._paper_balance:.2f}, "
                f"orders={len(self._paper_orders)}, "
                f"positions={len(self._paper_positions)})"
            )
        return f"OrderExecutor(mode='live')"
