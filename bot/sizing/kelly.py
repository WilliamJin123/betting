"""
Kelly Criterion Position Sizing for Binary Prediction Markets
==============================================================

The Kelly Criterion is a formula that tells you the optimal fraction of your
bankroll to bet, given that you believe you have an edge over the market.

The core idea:
    - A prediction market lists a contract at some price (e.g., $0.60).
    - That price *is* the market's implied probability (60% chance of YES).
    - If YOU believe the true probability is higher (say 75%), you have an EDGE.
    - Kelly tells you how much of your money to bet to grow your bankroll
      as fast as possible without going broke.

Why "quarter Kelly"?
    Full Kelly is mathematically optimal but assumes your probability estimates
    are perfect. They never are. Using a fraction of Kelly (like 1/4) sacrifices
    some growth rate for a LOT more safety. This is standard practice among
    professional bettors and traders.

Binary market Kelly formula:
    f* = (p_true - p_market) / (1 - p_market)

    Where:
        p_true   = your estimated probability the event happens
        p_market = the market price (implied probability)
        f*       = fraction of bankroll to bet

    If f* is positive, bet YES. If negative, bet NO.

Example:
    Market price = 0.40 (market says 40% chance)
    Your estimate = 0.55 (you think 55% chance)

    f* = (0.55 - 0.40) / (1 - 0.40) = 0.15 / 0.60 = 0.25

    Kelly says bet 25% of your bankroll on YES.
    Quarter Kelly says bet 0.25 * 25% = 6.25% of your bankroll.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_probability(value: float, name: str) -> None:
    """
    Ensure a probability is a float strictly between 0 and 1 (exclusive).

    We exclude 0 and 1 because:
        - A market at 0 or 1 is already resolved (no bet to make).
        - A personal estimate of 0 or 1 means you're 100% certain, which
          you never truly are.

    Raises:
        TypeError:  if value is not int or float.
        ValueError: if value is not in the open interval (0, 1).
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0.0 or value >= 1.0:
        raise ValueError(
            f"{name} must be strictly between 0 and 1 (exclusive), got {value}"
        )


def _validate_positive(value: float, name: str) -> None:
    """
    Ensure a value is a positive number.

    Raises:
        TypeError:  if value is not int or float.
        ValueError: if value is not > 0.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_fraction(value: float, name: str) -> None:
    """
    Ensure a value is between 0 and 1 inclusive.

    Raises:
        TypeError:  if value is not int or float.
        ValueError: if value is not in [0, 1].
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def edge(p_true: float, p_market: float) -> float:
    """
    Calculate your perceived edge over the market.

    Edge is simply how much your probability estimate differs from the
    market's implied probability.

        edge = p_true - p_market

    Interpretation:
        - Positive edge: you think YES is more likely than the market does.
          You would want to buy YES shares.
        - Negative edge: you think YES is less likely than the market does.
          You would want to buy NO shares (or sell YES if you hold them).
        - Zero edge: you agree with the market. No bet.

    Args:
        p_true:   Your estimated true probability of the event (0-1).
        p_market: Current market price / implied probability (0-1).

    Returns:
        The edge as a signed float. Positive = bet YES, negative = bet NO.

    Example:
        >>> edge(0.70, 0.55)
        0.15  # You think 15 percentage points more likely than the market
    """
    _validate_probability(p_true, "p_true")
    _validate_probability(p_market, "p_market")
    return p_true - p_market


def expected_value(p_true: float, p_market: float) -> float:
    """
    Calculate the expected value (EV) per dollar risked on a YES bet.

    EV answers: "On average, how much do I gain or lose per dollar I bet?"

    For a binary market where you buy YES at price p_market:
        - If you WIN (probability = p_true):  you receive $1, profit = 1 - p_market
        - If you LOSE (probability = 1 - p_true): you lose your stake = p_market

        EV = p_true * (1 - p_market) - (1 - p_true) * p_market
           = p_true - p_market

    This simplifies to the same thing as edge! That's not a coincidence:
    in a binary market, your edge IS your expected profit per dollar.

    Note: this is the EV for a YES bet. For a NO bet, flip p_true to
    (1 - p_true) and p_market to (1 - p_market) -- or just negate this value.

    Args:
        p_true:   Your estimated true probability of the event (0-1).
        p_market: Current market price / implied probability (0-1).

    Returns:
        Expected profit per dollar risked. Positive = profitable bet.

    Example:
        >>> expected_value(0.70, 0.55)
        0.15  # You expect to make $0.15 per $1 bet, on average
    """
    _validate_probability(p_true, "p_true")
    _validate_probability(p_market, "p_market")
    return p_true - p_market


def kelly_fraction(p_true: float, p_market: float) -> float:
    """
    Calculate the optimal Kelly fraction for a binary prediction market bet.

    This is the raw, un-adjusted Kelly fraction. It tells you what fraction
    of your bankroll to bet if your probability estimate is PERFECT (which
    it never is -- that's why we apply a multiplier later).

    The formula for binary prediction markets:

        f* = (p_true - p_market) / (1 - p_market)

    Derivation intuition:
        The numerator (p_true - p_market) is your edge -- how much better
        you think the true odds are vs. the market price.

        The denominator (1 - p_market) is the potential payout per dollar.
        If the market price is 0.40, you pay $0.40 and win $0.60 profit,
        so the payout ratio is 0.60.

        Kelly = edge / payout = how aggressively to bet relative to your
        edge and the odds you're getting.

    Interpretation of the output:
        - Positive f*: bet YES (buy YES shares)
        - Negative f*: bet NO (buy NO shares)
        - Zero f*: no edge, don't bet
        - f* > 1: extremely strong edge (rare, be suspicious of your estimate)

    Args:
        p_true:   Your estimated true probability of the event (0-1).
        p_market: Current market price / implied probability (0-1).

    Returns:
        Optimal fraction of bankroll to bet. Positive = YES, negative = NO.

    Examples:
        >>> kelly_fraction(0.70, 0.55)
        0.333...  # Bet 33% of bankroll on YES

        >>> kelly_fraction(0.30, 0.55)
        -0.555... # Bet on NO side (market overprices YES)
    """
    _validate_probability(p_true, "p_true")
    _validate_probability(p_market, "p_market")
    return (p_true - p_market) / (1.0 - p_market)


def time_discount(
    days_to_resolution: float,
    preferred_days: float = 7.0,
    power: float = 0.5,
) -> float:
    """
    Calculate a discount factor for how long your capital is locked up.

    The idea: a 10% edge that resolves in 7 days is much better than a 10%
    edge that resolves in 90 days, because in 7 days you get your money back
    and can reinvest it. Over a year, you could make ~52 weekly bets vs ~4
    quarterly bets.

    The formula:
        discount = min(1.0, (preferred_days / actual_days) ^ power)

    With default settings (preferred=7 days, power=0.5):
        - 1 day:  discount = 1.00 (capped, faster than preferred is fine)
        - 7 days: discount = 1.00 (the baseline)
        - 14 days: discount = 0.71
        - 30 days: discount = 0.48
        - 90 days: discount = 0.28
        - 180 days: discount = 0.20

    This discount is multiplied into the Kelly bet size, so a 90-day market
    gets ~28% of the bet you'd place on a 7-day market with the same edge.

    Args:
        days_to_resolution: How many days until the market resolves.
        preferred_days: The "ideal" resolution timeframe (discount = 1.0).
        power: How aggressively to penalize slow markets (0.5 = square root).

    Returns:
        Discount factor between 0 and 1.
    """
    if days_to_resolution <= 0:
        return 0.0  # Already expired or invalid
    if days_to_resolution <= preferred_days:
        return 1.0  # Faster than preferred is great, no penalty
    return min(1.0, (preferred_days / days_to_resolution) ** power)


def kelly_bet(
    p_true: float,
    p_market: float,
    bankroll: float,
    kelly_fraction_multiplier: float = 0.25,
    max_bet_fraction: float = 0.15,
    min_edge: float = 0.05,
    days_to_resolution: float | None = None,
) -> dict:
    """
    Calculate the actual bet to place, with all safety guardrails applied.

    This is the function you call in practice. It takes the raw Kelly
    fraction and applies four layers of protection:

    1. **Minimum edge filter** (min_edge):
       Don't bet at all unless your edge exceeds this threshold.
       Why? Small edges get eaten by fees, slippage, and estimation error.

    2. **Kelly multiplier** (kelly_fraction_multiplier):
       Multiply the raw Kelly fraction by this (default 0.25 = quarter Kelly).
       Why? Full Kelly assumes perfect probability estimates. Fractional
       Kelly dramatically reduces variance/drawdowns with only a small
       cost to long-run growth.

    3. **Time discount** (days_to_resolution):
       Scale down bet size for markets that take a long time to resolve.
       Why? Your capital is locked until the event happens. A 10% edge
       over 90 days is ~40% annualized, but 10% over 7 days is ~521%.
       Bet smaller on slow markets so capital stays available for faster ones.

    4. **Max bet cap** (max_bet_fraction):
       Never bet more than this fraction of your bankroll on one trade.
       Why? Even with Kelly, a single bad estimate shouldn't wreck you.
       With a $10 bankroll and 15% cap, max bet is $1.50.

    The function handles both YES and NO bets:
        - If your p_true > p_market, you think YES is underpriced -> bet YES.
        - If your p_true < p_market, you think YES is overpriced -> bet NO.
          For the NO side, the calculation is done by flipping the problem:
          treat it as a YES bet on the complementary event.

    Args:
        p_true:   Your estimated true probability (0-1).
        p_market: Current market price / implied probability (0-1).
        bankroll: Current bankroll in dollars. Must be positive.
        kelly_fraction_multiplier:
            Scale factor for Kelly. 1.0 = full Kelly, 0.5 = half Kelly,
            0.25 = quarter Kelly (recommended default). Must be in [0, 1].
        max_bet_fraction:
            Maximum fraction of bankroll to risk on any single bet.
            Must be in [0, 1]. Default 0.15 (15%).
        min_edge:
            Minimum absolute edge required to place a bet.
            Must be in [0, 1]. Default 0.05 (5 percentage points).
        days_to_resolution:
            Estimated days until the market resolves. If provided, the bet
            size is scaled down for slow-resolving markets. If None, no
            time discount is applied.

    Returns:
        A dictionary with:
            side:           "YES" or "NO" -- which side to bet on.
            size:           Dollar amount to bet (0.0 if no bet).
            edge:           Your raw edge (p_true - p_market). Signed.
            abs_edge:       Absolute value of edge.
            kelly_f:        Raw Kelly fraction (before multiplier/caps).
            adjusted_f:     Final fraction after multiplier and caps.
            time_discount:  Time discount factor applied (1.0 if not used).
            expected_value: Expected profit per dollar risked.
            reason:         Human-readable string explaining the decision.

    Examples:
        >>> kelly_bet(0.70, 0.55, bankroll=10.0)
        {
            'side': 'YES',
            'size': 0.83,         # ~8.3% of $10
            'edge': 0.15,
            'abs_edge': 0.15,
            'kelly_f': 0.333,
            'adjusted_f': 0.083,  # 0.333 * 0.25 = 0.083
            'time_discount': 1.0,
            'expected_value': 0.15,
            'reason': 'Bet YES: 8.3% edge, quarter-Kelly sizing'
        }

        >>> kelly_bet(0.70, 0.55, bankroll=10.0, days_to_resolution=60)
        {
            'side': 'YES',
            'size': 0.39,         # same edge, but scaled down for slow market
            ...
            'time_discount': 0.34,
        }

        >>> kelly_bet(0.52, 0.50, bankroll=10.0)
        {
            'side': 'YES',
            'size': 0.0,          # No bet! Edge (2%) < min_edge (5%)
            ...
            'reason': 'No bet: edge 2.0% below minimum 5.0%'
        }
    """
    # --- Validate inputs ---
    _validate_probability(p_true, "p_true")
    _validate_probability(p_market, "p_market")
    _validate_positive(bankroll, "bankroll")
    _validate_fraction(kelly_fraction_multiplier, "kelly_fraction_multiplier")
    _validate_fraction(max_bet_fraction, "max_bet_fraction")
    _validate_fraction(min_edge, "min_edge")

    # --- Determine direction ---
    raw_edge = p_true - p_market
    abs_edge = abs(raw_edge)

    if raw_edge >= 0:
        # We think YES is underpriced. Bet YES.
        side = "YES"
        # Kelly for YES: f* = (p_true - p_market) / (1 - p_market)
        raw_kelly = (p_true - p_market) / (1.0 - p_market)
        ev = p_true - p_market  # EV per dollar on YES
    else:
        # We think YES is overpriced. Bet NO.
        # Flip the problem: probability of NO = 1 - p_true,
        # market price of NO = 1 - p_market.
        side = "NO"
        p_no_true = 1.0 - p_true
        p_no_market = 1.0 - p_market
        raw_kelly = (p_no_true - p_no_market) / (1.0 - p_no_market)
        ev = p_no_true - p_no_market  # EV per dollar on NO

    # --- Compute time discount ---
    if days_to_resolution is not None and days_to_resolution > 0:
        td = time_discount(days_to_resolution)
    else:
        td = 1.0  # No time info available — don't penalize

    # --- Check minimum edge ---
    if abs_edge < min_edge:
        return {
            "side": side,
            "size": 0.0,
            "edge": round(raw_edge, 6),
            "abs_edge": round(abs_edge, 6),
            "kelly_f": round(raw_kelly, 6),
            "adjusted_f": 0.0,
            "time_discount": round(td, 4),
            "expected_value": round(ev, 6),
            "reason": (
                f"No bet: edge {abs_edge * 100:.1f}% "
                f"below minimum {min_edge * 100:.1f}%"
            ),
        }

    # --- Apply Kelly multiplier ---
    adjusted_f = raw_kelly * kelly_fraction_multiplier

    # --- Apply time discount ---
    adjusted_f *= td

    # --- Apply max bet cap ---
    adjusted_f = min(adjusted_f, max_bet_fraction)

    # --- Calculate dollar amount ---
    size = round(adjusted_f * bankroll, 2)

    # Don't return negative sizes (shouldn't happen after our logic, but
    # be defensive).
    size = max(size, 0.0)

    time_note = ""
    if days_to_resolution is not None and td < 1.0:
        time_note = f", time discount {td:.0%} ({days_to_resolution:.0f}d)"

    reason = (
        f"Bet {side}: {abs_edge * 100:.1f}% edge, "
        f"Kelly {raw_kelly * 100:.1f}% -> "
        f"adjusted {adjusted_f * 100:.1f}% of bankroll "
        f"(${size:.2f}){time_note}"
    )

    return {
        "side": side,
        "size": size,
        "edge": round(raw_edge, 6),
        "abs_edge": round(abs_edge, 6),
        "kelly_f": round(raw_kelly, 6),
        "adjusted_f": round(adjusted_f, 6),
        "time_discount": round(td, 4),
        "expected_value": round(ev, 6),
        "reason": reason,
    }
