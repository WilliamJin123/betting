"""
Bot Configuration
=================

Central configuration for the Polymarket prediction market bot.

All tunable parameters live here so you never have to dig through code
to change a setting. When the bot starts, it imports from this file.

How to think about these settings:
    - BANKROLL:   How much real money the bot manages.
    - KELLY:      How aggressively to size bets (lower = safer).
    - API:        Where to talk to Polymarket's servers.
    - SCANNER:    Filters for which markets are worth looking at.
    - LOGGING:    Where trade records are saved.
"""

import os

# ---------------------------------------------------------------------------
# Bankroll
# ---------------------------------------------------------------------------

INITIAL_BANKROLL: float = 10.0
"""
Starting bankroll in USD.

We're beginning with $10 to learn the system with minimal risk.
As confidence grows, this can be increased. The bot tracks its own
running bankroll from trade history, but this is the seed value.
"""

# ---------------------------------------------------------------------------
# Kelly Criterion Settings
# ---------------------------------------------------------------------------

KELLY_MULTIPLIER: float = 0.25
"""
Fraction of Kelly to use (0.25 = quarter Kelly).

Full Kelly (1.0) maximizes long-run growth rate but has brutal drawdowns.
Quarter Kelly gives ~75% of the growth rate with drastically less variance.
Most professionals use somewhere between 0.1 and 0.5.

Rule of thumb:
    - 1.0  = full Kelly (aggressive, not recommended)
    - 0.5  = half Kelly (moderate)
    - 0.25 = quarter Kelly (conservative, our default)
    - 0.1  = tenth Kelly (very conservative)
"""

MAX_BET_FRACTION: float = 0.15
"""
Maximum fraction of bankroll to bet on any single market.

Even if Kelly says bet 50%, this caps it at 15%. With a $10 bankroll,
the max single bet is $1.50. This is a hard safety rail.
"""

MIN_EDGE: float = 0.05
"""
Minimum edge (as a decimal) required to place a bet.

0.05 = 5 percentage points. If you think a market at 40% should be 44%,
that's only a 4% edge -- below our threshold. We skip it because:
    1. Small edges get wiped out by trading fees (~2% on Polymarket).
    2. Our probability estimates have error bars. A 3% "edge" might
       actually be a 0% edge or even negative.
    3. Transaction costs (gas fees on Polygon) eat into tiny bets.
"""

# ---------------------------------------------------------------------------
# Polymarket API Endpoints
# ---------------------------------------------------------------------------

GAMMA_API_BASE: str = "https://gamma-api.polymarket.com"
"""
Gamma API: used for discovering markets, reading metadata, and fetching
historical data. This is the "read" API -- it doesn't place trades.

Docs: https://docs.polymarket.com/
"""

CLOB_API_BASE: str = "https://clob.polymarket.com"
"""
CLOB (Central Limit Order Book) API: used for placing and managing orders.
This is the "write" API -- it's where actual trading happens.

Requires authentication with a Polymarket API key.
Docs: https://docs.polymarket.com/
"""

# ---------------------------------------------------------------------------
# Market Scanner Settings
# ---------------------------------------------------------------------------

MIN_VOLUME: int = 1000
"""
Minimum total trading volume (in USD) for a market to be considered.

Markets with very low volume are illiquid -- you can't get in or out
easily, and the prices may not reflect real information. $1000 is a
conservative floor.
"""

MIN_LIQUIDITY: int = 500
"""
Minimum current liquidity (in USD) in the order book.

Liquidity = how much money is sitting in open orders. Low liquidity
means your trade will move the price against you (slippage). $500
ensures there's at least some depth.
"""

STALE_HOURS: int = 24
"""
Flag markets where the price hasn't changed in this many hours.

A market with a frozen price might be:
    - Already effectively resolved (everyone agrees)
    - Abandoned / no one is trading it
    - Broken in some way

These are usually not worth betting on.
"""

# ---------------------------------------------------------------------------
# Time-to-Resolution Settings
# ---------------------------------------------------------------------------

PREFERRED_RESOLUTION_DAYS: float = 7.0
"""
Ideal market resolution time in days.

Markets resolving around this timeframe get full Kelly sizing.
Faster markets are slightly better (capital frees up sooner).
Slower markets get their bet size scaled down because your capital
is locked up longer — the same edge is worth less if you can't
reinvest for 6 months vs 1 week.
"""

MAX_RESOLUTION_DAYS: float = 90.0
"""
Markets resolving beyond this many days get heavily penalized.

A 10% edge that takes 90 days to resolve is only ~40% annualized.
The same 10% edge resolving in 7 days is ~521% annualized.
We don't refuse to bet on slow markets, but we scale down significantly.
"""

TIME_DISCOUNT_POWER: float = 0.5
"""
How aggressively to penalize slow-resolving markets.

The time discount formula is:
    discount = (preferred_days / actual_days) ^ power

With power=0.5 (square root):
    - 7-day market: discount = 1.0 (no penalty)
    - 14-day market: discount = 0.71
    - 30-day market: discount = 0.48
    - 90-day market: discount = 0.28

Higher power = more aggressive penalty for slow markets.
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR: str = "logs"
"""
Directory where log files are stored, relative to the project root.
"""

TRADE_LOG_FILE: str = "trades.jsonl"
"""
File where individual trade records are appended, one JSON object per line.

JSONL format (JSON Lines) means each line is a valid JSON object.
This makes it easy to:
    - Append new trades without reading the whole file
    - Process trades line by line for analysis
    - Load into pandas with pd.read_json('trades.jsonl', lines=True)
"""

# ---------------------------------------------------------------------------
# Derived paths (computed, not settings you change)
# ---------------------------------------------------------------------------

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Absolute path to the project root directory."""

TRADE_LOG_PATH: str = os.path.join(PROJECT_ROOT, LOG_DIR, TRADE_LOG_FILE)
"""Full path to the trade log file."""
