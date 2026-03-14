"""
Scanner Data Models
===================

Data models for the market scanner's output. Each scan produces a list of
ScanResult objects that describe potential opportunities found in the market.

These are intentionally simple dataclasses -- the scanner fills them in,
and downstream code (ranking, alerting, eventual trading) consumes them.
"""

from dataclasses import dataclass, field
from typing import Any

from bot.polymarket.models import Market


@dataclass
class ScanResult:
    """
    One opportunity flagged by the scanner.

    Fields:
        market:           The Market object this opportunity was found in.
        opportunity_type: What kind of opportunity this is. One of:
                          - "stale"         : price appears frozen / under-traded
                          - "arb"           : multi-outcome prices don't sum to 1.00
                          - "wide_spread"   : bid-ask spread is unusually wide
                          - "manual_review" : doesn't fit a clean category but
                                              looks interesting (e.g., high volume
                                              spike, near-expiry with uncertainty)
        score:            Attractiveness ranking from 0.0 (worthless) to 1.0
                          (extremely attractive). Used by rank_opportunities()
                          to sort results.
        details:          Type-specific metadata. Examples:
                          - arb:         {"price_sum": 0.94, "gap": 0.06,
                                          "profit_per_dollar": 0.064}
                          - wide_spread: {"spread": 0.08, "best_bid": 0.45,
                                          "best_ask": 0.53}
                          - stale:       {"volume_ratio_24h": 0.002,
                                          "price_near_round": True,
                                          "hours_to_expiry": 48}
        timestamp:        ISO 8601 string of when this scan was performed.
    """
    market: Market
    opportunity_type: str
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __repr__(self) -> str:
        return (
            f"ScanResult(type={self.opportunity_type!r}, "
            f"score={self.score:.3f}, "
            f"market={self.market.question[:60]!r})"
        )
