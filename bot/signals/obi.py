"""
Order Book Imbalance (OBI) Module
==================================

Computes entry timing signals from order book data. The core idea:

    The order book shows all pending buy orders (bids) and sell orders (asks).
    If there are way more bids than asks, buying pressure is high and the price
    is likely being pushed UP right now. If we want to buy, that's a bad time
    (we're buying into a pump). Conversely, if selling pressure dominates,
    the price is being pushed DOWN -- a better time to buy (we get a discount).

Three metrics:

    1. OBI (Order Book Imbalance): normalized difference between bid and ask volume.
       Ranges from -1 (all selling pressure) to +1 (all buying pressure).

    2. VAMP (Volume-Adjusted Mid Price): a "fairer" mid-price that accounts for
       order sizes, not just the best bid/ask prices. When the book is lopsided,
       VAMP diverges from the simple midpoint -- and VAMP is more accurate.

    3. Imbalance Ratio: bid_volume / total_volume. >0.65 predicts price increase
       within 15-30 minutes (~58% accuracy per the research).

Usage:
    from bot.signals.obi import compute_obi, should_wait_for_entry

    book = client.get_order_book(token_id)
    signal = should_wait_for_entry(book, side="BUY")

    if signal["signal"] == "buy_now":
        place_order(...)
    elif signal["signal"] == "wait":
        print(f"Wait {signal['suggested_wait_minutes']}min: {signal['reason']}")
"""

from bot.polymarket.models import OrderBook


# ---------------------------------------------------------------------------
# Thresholds (derived from X posts / microstructure research)
# ---------------------------------------------------------------------------

OBI_STRONG_BUY_THRESHOLD = -0.3
"""
OBI below this means significant selling pressure. Price is being pushed down,
which is a good time to buy (you're getting a discount).
"""

OBI_WAIT_THRESHOLD = 0.5
"""
OBI above this means strong buying pressure. Price is being pushed up right now.
If you buy now, you're buying into a pump and will likely overpay. Wait for it
to settle.
"""

IR_BULLISH_THRESHOLD = 0.65
"""
Imbalance Ratio above 0.65 predicts a price increase within 15-30 minutes
(~58% accuracy). This means bids heavily outweigh asks.
"""

IR_BEARISH_THRESHOLD = 0.35
"""
Imbalance Ratio below 0.35 predicts a price decrease. Asks heavily outweigh bids.
"""


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_obi(order_book: OrderBook) -> float:
    """
    Order Book Imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

    Returns a value between -1.0 and +1.0:
        +1.0 = all buying pressure (price likely to rise)
        -1.0 = all selling pressure (price likely to fall)
         0.0 = balanced

    If the book is completely empty (no bids AND no asks), returns 0.0 (no signal).

    We use raw share volume (sum of sizes), not dollar volume, because in
    prediction markets all prices are 0-1 and what matters is the number
    of shares people want to trade at each level.
    """
    total_bid = sum(level.size for level in order_book.bids)
    total_ask = sum(level.size for level in order_book.asks)

    total = total_bid + total_ask
    if total == 0:
        return 0.0

    return (total_bid - total_ask) / total


def compute_vamp(order_book: OrderBook) -> float | None:
    """
    Volume-Adjusted Mid Price (VAMP).

    Formula:
        VAMP = (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)

    Where P_bid/P_ask are the best bid/ask prices and Q_bid/Q_ask are the
    total volumes on each side.

    Why this is useful:
        The simple midpoint (bid + ask) / 2 treats both sides equally. But if
        there are 1000 shares on the bid side and only 10 on the ask side,
        the "fair" price is much closer to the bid. VAMP captures this.

    Returns None if the book has no bids or no asks (can't compute VAMP
    without both sides).
    """
    if not order_book.bids or not order_book.asks:
        return None

    best_bid = order_book.best_bid
    best_ask = order_book.best_ask

    if best_bid <= 0 or best_ask <= 0:
        return None

    total_bid_vol = sum(level.size for level in order_book.bids)
    total_ask_vol = sum(level.size for level in order_book.asks)

    total_vol = total_bid_vol + total_ask_vol
    if total_vol == 0:
        return None

    return (best_bid * total_ask_vol + best_ask * total_bid_vol) / total_vol


def compute_imbalance_ratio(order_book: OrderBook) -> float:
    """
    Imbalance Ratio = bid_volume / (bid_volume + ask_volume)

    Ranges from 0.0 to 1.0:
        > 0.65 predicts price increase within 15-30 minutes
        < 0.35 predicts price decrease
        ~0.50 = balanced, no directional prediction

    Returns 0.5 (neutral) if the book is empty.
    """
    total_bid = sum(level.size for level in order_book.bids)
    total_ask = sum(level.size for level in order_book.asks)

    total = total_bid + total_ask
    if total == 0:
        return 0.5

    return total_bid / total


# ---------------------------------------------------------------------------
# Entry Signal
# ---------------------------------------------------------------------------

def should_wait_for_entry(order_book: OrderBook, side: str = "BUY") -> dict:
    """
    Given we want to BUY a token (either YES or NO), should we wait for a
    better price?

    Logic:
        - If we want to BUY and OBI is very positive (lots of buying pressure),
          the price is being pushed UP right now. We'd be buying at an inflated
          price. Signal: WAIT for the buying pressure to subside.

        - If we want to BUY and OBI is negative (selling pressure), the price
          is being pushed DOWN. This means we'd buy at a discount. Signal:
          STRONG_BUY (good time to enter).

        - If balanced (OBI near zero), there's no strong pressure either way.
          Signal: BUY_NOW (proceed normally).

    Args:
        order_book: Current order book snapshot from the CLOB API.
        side: "BUY" (we always buy tokens on Polymarket, whether YES or NO).

    Returns:
        dict with keys:
            signal:                 "buy_now" | "wait" | "strong_buy"
            obi:                    float, the raw OBI value
            imbalance_ratio:        float, bid_vol / total_vol
            vamp:                   float or None
            midpoint:               float, simple (bid+ask)/2 from the book
            reason:                 str, human-readable explanation
            suggested_wait_minutes: int (0 if buy_now/strong_buy, 15-30 if wait)
    """
    obi = compute_obi(order_book)
    ir = compute_imbalance_ratio(order_book)
    vamp = compute_vamp(order_book)

    # Default: proceed normally
    signal = "buy_now"
    reason = "Order book is balanced -- no strong pressure either way."
    wait_minutes = 0

    if obi >= OBI_WAIT_THRESHOLD:
        # Strong buying pressure -- price is being pumped up
        signal = "wait"
        reason = (
            f"Strong buying pressure (OBI={obi:+.2f}). Price is being pushed up. "
            f"Wait for it to settle before buying."
        )
        # Suggest longer wait when pressure is more extreme
        if obi >= 0.7:
            wait_minutes = 30
        else:
            wait_minutes = 15

    elif obi <= OBI_STRONG_BUY_THRESHOLD:
        # Selling pressure -- price is being pushed down = discount
        signal = "strong_buy"
        reason = (
            f"Selling pressure (OBI={obi:+.2f}). Price is being pushed down -- "
            f"good entry point (buying at a discount)."
        )
        wait_minutes = 0

    # Refine with imbalance ratio if OBI was borderline
    elif ir >= IR_BULLISH_THRESHOLD:
        signal = "wait"
        reason = (
            f"Imbalance ratio is high ({ir:.2f} > {IR_BULLISH_THRESHOLD}). "
            f"Bids heavily outweigh asks -- price likely rising. Consider waiting."
        )
        wait_minutes = 15

    elif ir <= IR_BEARISH_THRESHOLD:
        signal = "strong_buy"
        reason = (
            f"Imbalance ratio is low ({ir:.2f} < {IR_BEARISH_THRESHOLD}). "
            f"Asks outweigh bids -- price likely falling. Good entry."
        )
        wait_minutes = 0

    return {
        "signal": signal,
        "obi": obi,
        "imbalance_ratio": ir,
        "vamp": vamp,
        "midpoint": order_book.midpoint,
        "reason": reason,
        "suggested_wait_minutes": wait_minutes,
    }
