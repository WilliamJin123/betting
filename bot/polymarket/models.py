"""
Polymarket Data Models
======================

Plain dataclass models for the data we get back from Polymarket's APIs.

Why dataclasses instead of raw dicts?
    - Autocompletion in your editor (you can type market.question instead of market["question"])
    - Typos become errors instead of silent bugs
    - Clear documentation of what fields exist and what types they are

Why not Pydantic?
    - We're keeping dependencies minimal for now. Dataclasses are built-in.
    - If we later need validation (e.g., "price must be between 0 and 1"),
      we can swap to Pydantic. For read-only market data, dataclasses are fine.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Token — a single outcome token within a market
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """
    One side of a binary market (e.g., "Yes" or "No").

    Each outcome in a Polymarket market has its own token with a unique ID.
    This token_id is what you pass to the CLOB API for prices and order books.

    Fields:
        token_id:  Huge numeric string that uniquely identifies this token on-chain.
        outcome:   Human-readable label like "Yes", "No", "Trump", "Biden", etc.
        price:     Current price from 0.00 to 1.00. This IS the implied probability.
                   A price of 0.65 means the market thinks there's a 65% chance.
        winner:    True if this outcome has been resolved as the winner. Only
                   meaningful for closed/resolved markets.
    """
    token_id: str
    outcome: str
    price: float = 0.0
    winner: bool = False


# ---------------------------------------------------------------------------
# Market — a prediction market (the main thing we care about)
# ---------------------------------------------------------------------------

@dataclass
class Market:
    """
    A single prediction market on Polymarket.

    This combines data from both the Gamma API (metadata like question, dates,
    volume) and the CLOB API (live prices, order book status).

    Key concepts:
        - condition_id: The market's unique identifier (used in CLOB API).
        - tokens: Each market has outcome tokens. A binary market has 2 tokens
          (Yes/No). The token prices always sum to ~1.00.
        - volume: Total USD traded. More volume = more reliable prices.
        - liquidity: USD sitting in the order book right now. More liquidity =
          less slippage when you trade.

    Fields:
        condition_id:   Hex string identifying this market in the CLOB API.
        question:       The human-readable question, e.g. "Will Bitcoin hit $100k by 2025?"
        slug:           URL-friendly version of the question (for building Polymarket URLs).
        description:    Longer text explaining resolution criteria.
        outcomes:       List of outcome labels like ["Yes", "No"].
        outcome_prices: List of prices corresponding to each outcome.
        tokens:         List of Token objects with full token details.
        end_date:       ISO 8601 string for when this market closes.
        active:         Is this market currently active (accepting trades)?
        closed:         Has this market been resolved/closed?
        volume:         Total USD traded across the life of this market.
        liquidity:      Current USD in the order book.
        volume_24h:     USD traded in the last 24 hours.
        spread:         Difference between best ask and best bid.
        last_trade_price: Price of the most recent trade.
        best_bid:       Highest price someone is willing to buy at.
        best_ask:       Lowest price someone is willing to sell at.
        min_order_size: Minimum trade size in USD.
        tick_size:      Smallest price increment (e.g., 0.01 = penny increments).
        neg_risk:       Whether this is a "negative risk" market (multi-outcome markets
                        where tokens are structured differently).
    """
    condition_id: str
    question: str = ""
    slug: str = ""
    description: str = ""
    outcomes: list[str] = field(default_factory=list)
    outcome_prices: list[float] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    end_date: str = ""
    active: bool = False
    closed: bool = False
    volume: float = 0.0
    liquidity: float = 0.0
    volume_24h: float = 0.0
    spread: float = 0.0
    last_trade_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    min_order_size: float = 0.0
    tick_size: float = 0.01
    neg_risk: bool = False

    @property
    def yes_price(self) -> Optional[float]:
        """
        Convenience: price of the "Yes" token, or the first outcome.

        Returns None if no outcomes/prices are available.
        """
        if self.outcome_prices:
            return self.outcome_prices[0]
        return None

    @property
    def no_price(self) -> Optional[float]:
        """
        Convenience: price of the "No" token, or the second outcome.

        Returns None if fewer than 2 outcomes.
        """
        if len(self.outcome_prices) >= 2:
            return self.outcome_prices[1]
        return None

    @property
    def url(self) -> str:
        """Full Polymarket URL for this market."""
        if self.slug:
            return f"https://polymarket.com/event/{self.slug}"
        return ""

    @property
    def yes_token_id(self) -> Optional[str]:
        """Token ID for the first outcome (usually Yes). Needed for CLOB API calls."""
        if self.tokens:
            return self.tokens[0].token_id
        return None

    @property
    def no_token_id(self) -> Optional[str]:
        """Token ID for the second outcome (usually No). Needed for CLOB API calls."""
        if len(self.tokens) >= 2:
            return self.tokens[1].token_id
        return None


# ---------------------------------------------------------------------------
# OrderBookLevel — a single price level in the order book
# ---------------------------------------------------------------------------

@dataclass
class OrderBookLevel:
    """
    A single price level in the order book.

    Think of it like a row in this table:

        Price  |  Size
        -------|-------
        $0.65  |  150    <-- someone wants to buy 150 shares at $0.65

    Fields:
        price:  The price for this level (0.00 to 1.00).
        size:   How many shares are available at this price.
    """
    price: float
    size: float


# ---------------------------------------------------------------------------
# OrderBook — the full order book snapshot for one token
# ---------------------------------------------------------------------------

@dataclass
class OrderBook:
    """
    A snapshot of the order book for a single outcome token.

    The order book shows all open buy orders (bids) and sell orders (asks).

    How to read it:
        - Bids are sorted highest-first (best bid at top). These are people
          willing to BUY at that price.
        - Asks are sorted lowest-first (best ask at top). These are people
          willing to SELL at that price.
        - The spread = best_ask - best_bid. Tighter spread = more liquid market.

    Example:
        Asks: $0.67 (50 shares), $0.68 (100 shares), $0.70 (200 shares)
        Bids: $0.65 (150 shares), $0.64 (80 shares), $0.60 (300 shares)
        Spread: $0.67 - $0.65 = $0.02

    Fields:
        market:          Condition ID (hex string) of the market.
        asset_id:        Token ID for this specific outcome.
        bids:            Buy orders, sorted best (highest) first.
        asks:            Sell orders, sorted best (lowest) first.
        spread:          Difference between best ask and best bid.
        midpoint:        Average of best bid and best ask.
        best_bid:        Highest bid price (or 0 if no bids).
        best_ask:        Lowest ask price (or 0 if no asks).
        last_trade_price: Price of the most recent trade.
        min_order_size:  Minimum trade size.
        tick_size:       Smallest price increment.
        timestamp:       When this snapshot was taken (ms since epoch).
    """
    market: str = ""
    asset_id: str = ""
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    spread: float = 0.0
    midpoint: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_trade_price: float = 0.0
    min_order_size: float = 0.0
    tick_size: float = 0.01
    timestamp: str = ""

    @property
    def bid_depth(self) -> float:
        """Total USD available on the bid side (sum of price * size for all bids)."""
        return sum(level.price * level.size for level in self.bids)

    @property
    def ask_depth(self) -> float:
        """Total USD available on the ask side (sum of price * size for all asks)."""
        return sum(level.price * level.size for level in self.asks)


# ---------------------------------------------------------------------------
# Position — tracks a position we hold (for future use)
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    A position we hold in a market.

    This isn't used in Phase 1 (read-only), but it's here so we have the
    shape ready when we start trading.

    Fields:
        market_id:    Condition ID of the market.
        token_id:     Which outcome token we hold.
        side:         "YES" or "NO" (which outcome we bet on).
        size:         How many shares we hold.
        entry_price:  Average price we paid per share.
        current_price: What the shares are worth now.
    """
    market_id: str
    token_id: str
    side: str
    size: float
    entry_price: float
    current_price: float = 0.0

    @property
    def cost_basis(self) -> float:
        """Total USD we spent to acquire this position."""
        return self.size * self.entry_price

    @property
    def market_value(self) -> float:
        """What the position is worth right now."""
        return self.size * self.current_price

    @property
    def pnl(self) -> float:
        """Profit/loss in USD (positive = profit)."""
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        """Profit/loss as a percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return (self.pnl / self.cost_basis) * 100
