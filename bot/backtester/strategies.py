"""
Baseline Backtesting Strategies
================================

Simple, intentionally naive strategies for verifying the backtesting harness.
These are NOT meant to make money — they exist to sanity-check the engine:

    - always_fifty:   Always predict 50% -> should produce ~0 P&L
    - always_yes_60:  Always predict 60% YES -> reveals YES-outcome base rate
    - keyword_heuristic: Simple keyword matching -> tests the plumbing

A real strategy would use an LLM, statistical model, or domain expertise.
"""

from __future__ import annotations

import re
from typing import Optional

from bot.backtester.data import ResolvedMarket


def always_fifty(market: ResolvedMarket) -> float:
    """
    Always predict 50/50.

    Expected behavior: with a 50% market price proxy, this produces zero
    edge and Kelly should skip every bet.  Useful as a "null hypothesis"
    baseline — if this strategy somehow makes money, something is wrong
    with the backtester.
    """
    return 0.5


def always_yes_60(market: ResolvedMarket) -> float:
    """
    Always predict 60% YES.

    This has a naive bullish bias — it always thinks YES is more likely
    than 50%.  Against the 50% market proxy, Kelly will always bet YES.
    The P&L reveals the base rate: if YES wins more than 50% of the time,
    this strategy profits.

    On Polymarket, YES outcomes may indeed be slightly more common than 50%
    because markets are often phrased as "Will X happen?" where X is a
    plausible event.
    """
    return 0.6


def keyword_heuristic(market: ResolvedMarket) -> Optional[float]:
    """
    Simple keyword-based heuristic for estimating YES probability.

    Rules:
    - Questions containing "Bitcoin" and "above" with a dollar amount:
      return a rough probability that decreases with the target price.
    - Questions containing "win" or "winner": return 0.45
      (slight underdog bias — markets tend to overvalue favorites).
    - Questions about elections or "president": return 0.50
      (no edge, basically a pass — but demonstrates category matching).
    - Everything else: return None (no opinion).
    """
    q = market.question.lower()

    # Bitcoin price targets
    if "bitcoin" in q and "above" in q:
        # Try to extract a dollar amount
        price_match = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*[k]?', q)
        if price_match:
            try:
                target_str = price_match.group(1).replace(",", "")
                target = float(target_str)
                # If the match ended with 'k', multiply by 1000
                if 'k' in q[price_match.start():price_match.end() + 2].lower():
                    target *= 1000

                # Rough heuristic: lower targets more likely
                # BTC has been ~$30k-$100k range recently
                if target < 30_000:
                    return 0.80
                elif target < 60_000:
                    return 0.60
                elif target < 100_000:
                    return 0.45
                elif target < 150_000:
                    return 0.30
                else:
                    return 0.15
            except ValueError:
                pass
        return 0.50  # Bitcoin question but couldn't parse target

    # Competitions / sports / elections with "win"
    if "win" in q or "winner" in q:
        return 0.45  # Slight underdog bias

    # Elections
    if "president" in q or "election" in q:
        return 0.50  # No edge on politics — punt

    # No opinion on everything else
    return None


# ---------------------------------------------------------------------------
# Strategy registry for CLI access
# ---------------------------------------------------------------------------

STRATEGIES = {
    "always_fifty": always_fifty,
    "always_yes_60": always_yes_60,
    "keyword": keyword_heuristic,
}


def get_strategy(name: str):
    """Look up a strategy by name. Raises KeyError if not found."""
    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown strategy '{name}'. "
            f"Available: {', '.join(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]
