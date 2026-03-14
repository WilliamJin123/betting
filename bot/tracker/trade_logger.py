"""
Trade Logger — Records every bet and tracks profit/loss over time.

WHY THIS EXISTS:
    You can't improve what you don't measure.  Every bet we place gets
    recorded as a single JSON line in a ``.jsonl`` file (JSON Lines format).
    When a market resolves we update that line with the outcome and P&L.

    This gives us a permanent, append-friendly ledger we can analyse later
    to answer questions like:
        - Are we actually profitable?
        - Are our probability estimates well-calibrated?
        - What's our win rate? Average edge?

FILE FORMAT — JSONL:
    Each line in the file is a standalone JSON object.  This is better than
    a single big JSON array because:
        1. We can append without reading the whole file
        2. If one line is corrupt the rest are fine
        3. Easy to stream / process line-by-line

P&L CALCULATION FOR PREDICTION MARKETS:
    Prediction market shares pay $1 if the event happens, $0 if not.

    When you buy YES shares at price ``p``:
        - You spend ``size_usd`` dollars
        - You receive ``size_usd / p`` shares (each costs ``p`` dollars)
        - If you WIN:  each share pays $1  ->  payout = size_usd / p
                       profit = payout - size_usd = size_usd * (1 - p) / p
        - If you LOSE: shares pay $0       ->  profit = -size_usd

    When you buy NO shares at price ``p`` (where p is your entry price for NO):
        - Same maths, just the other direction. You're betting the event
          does NOT happen.  NO shares pay $1 if the event doesn't happen.

    Example:
        Buy YES at $0.60, spending $10
        Shares = 10 / 0.60 = 16.67 shares
        WIN  -> payout = 16.67 * $1 = $16.67, profit = +$6.67
        LOSS -> payout = 0, profit = -$10.00
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.tracker.calibration import compute_calibration, print_calibration


# ---------------------------------------------------------------------------
# Default log location — ``<project_root>/logs/trades.jsonl``
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # bot/tracker -> bot -> project root
DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "trades.jsonl"


class TradeLogger:
    """Append-only trade ledger backed by a JSONL file.

    Parameters
    ----------
    log_path : Path or str, optional
        Where to store the JSONL file.  Defaults to ``<project_root>/logs/trades.jsonl``.
    starting_bankroll : float
        The amount of money you started with.  Used in the summary to
        compute current bankroll (starting_bankroll + total_pnl).
    """

    def __init__(
        self,
        log_path: Path | str | None = None,
        starting_bankroll: float = 10.0,
    ) -> None:
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_FILE
        self.starting_bankroll = starting_bankroll

        # Make sure the parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the file if it doesn't exist
        if not self.log_path.exists():
            self.log_path.touch()

    # ------------------------------------------------------------------
    # Core I/O helpers
    # ------------------------------------------------------------------

    def _read_all_trades(self) -> list[dict[str, Any]]:
        """Read every trade record from the JSONL file.

        Returns a list of dicts, one per trade.  Blank lines are skipped.
        """
        trades: list[dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
        return trades

    def _write_all_trades(self, trades: list[dict[str, Any]]) -> None:
        """Rewrite the entire JSONL file from a list of trade dicts.

        This is used when we need to *update* an existing record (e.g.
        resolving a trade).  For normal appends use ``_append_trade``.
        """
        with open(self.log_path, "w", encoding="utf-8") as f:
            for trade in trades:
                f.write(json.dumps(trade) + "\n")

    def _append_trade(self, trade: dict[str, Any]) -> None:
        """Append a single trade record as a new line."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_trade(
        self,
        market_id: str,
        market_question: str,
        side: str,
        size_usd: float,
        entry_price: float,
        p_estimated: float,
        p_market: float,
        edge: float,
        kelly_fraction: float,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Record a new bet.

        Parameters
        ----------
        market_id : str
            Polymarket's unique identifier for this market / condition.
        market_question : str
            Human-readable text, e.g. "Will BTC hit $100k by June?".
        side : str
            ``"YES"`` or ``"NO"`` — which outcome you're betting on.
        size_usd : float
            Dollar amount wagered.
        entry_price : float
            Price paid per share, between 0 and 1.
            For a YES bet, this is the YES price.  For NO, the NO price.
        p_estimated : float
            Our model's estimated true probability of the YES outcome.
        p_market : float
            Market's implied probability at the time of the bet.
            For a YES bet this equals the YES price; for NO, 1 - NO_price.
        edge : float
            ``p_estimated - p_market``.  Positive means we think the market
            is underpricing the event.
        kelly_fraction : float
            The Kelly criterion fraction we used for sizing.
        timestamp : str, optional
            ISO-format timestamp.  Defaults to the current UTC time.

        Returns
        -------
        dict
            The full trade record that was written.
        """
        if side not in ("YES", "NO"):
            raise ValueError(f"side must be 'YES' or 'NO', got {side!r}")
        if not (0 < entry_price < 1):
            raise ValueError(f"entry_price must be between 0 and 1 (exclusive), got {entry_price}")
        if size_usd <= 0:
            raise ValueError(f"size_usd must be positive, got {size_usd}")

        record: dict[str, Any] = {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "market_question": market_question,
            "side": side,
            "size_usd": round(size_usd, 4),
            "entry_price": round(entry_price, 4),
            "shares": round(size_usd / entry_price, 4),
            "p_estimated": round(p_estimated, 4),
            "p_market": round(p_market, 4),
            "edge": round(edge, 4),
            "kelly_fraction": round(kelly_fraction, 4),
            "outcome": None,
            "pnl": None,
            "resolved_at": None,
        }

        self._append_trade(record)
        return record

    def resolve_trade(
        self,
        market_id: str,
        outcome: bool,
        resolved_at: str | None = None,
        timestamp: str | None = None,
    ) -> list[dict[str, Any]]:
        """Mark one or more trades on ``market_id`` as resolved.

        A single market can have multiple bets (placed at different times).
        This method resolves ALL open bets on that market in one go.

        Parameters
        ----------
        market_id : str
            The market identifier whose trades should be resolved.
        outcome : bool
            ``True`` if the YES outcome happened, ``False`` if NO happened.
        resolved_at : str, optional
            ISO-format timestamp of the resolution.  Defaults to now (UTC).
        timestamp : str, optional
            *Deprecated alias for resolved_at*.  If both are given,
            ``resolved_at`` takes priority.

        Returns
        -------
        list[dict]
            The trade records that were updated.

        How P&L is calculated
        ---------------------
        For a YES bet:
            WIN  (outcome=True):  pnl = (size_usd / entry_price) - size_usd
            LOSS (outcome=False): pnl = -size_usd

        For a NO bet:
            WIN  (outcome=False): pnl = (size_usd / entry_price) - size_usd
            LOSS (outcome=True):  pnl = -size_usd
        """
        resolve_ts = resolved_at or timestamp or datetime.now(timezone.utc).isoformat()

        trades = self._read_all_trades()
        updated: list[dict[str, Any]] = []

        for trade in trades:
            if trade["market_id"] != market_id:
                continue
            if trade["outcome"] is not None:
                # Already resolved — don't touch it
                continue

            # Determine if this particular bet won
            side = trade["side"]
            if side == "YES":
                bet_won = outcome  # YES bet wins when the event happens
            else:
                bet_won = not outcome  # NO bet wins when the event doesn't happen

            size = trade["size_usd"]
            price = trade["entry_price"]

            if bet_won:
                # Shares pay $1 each
                payout = size / price
                pnl = payout - size
                trade["outcome"] = "WIN"
            else:
                # Shares pay $0
                pnl = -size
                trade["outcome"] = "LOSS"

            trade["pnl"] = round(pnl, 4)
            trade["resolved_at"] = resolve_ts
            updated.append(trade)

        if not updated:
            raise ValueError(
                f"No open trades found for market_id={market_id!r}. "
                f"Either no bets were placed on this market or they were already resolved."
            )

        self._write_all_trades(trades)
        return updated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Return all trades that have not been resolved yet."""
        return [t for t in self._read_all_trades() if t["outcome"] is None]

    def get_resolved_trades(self) -> list[dict[str, Any]]:
        """Return all trades that have been resolved (WIN or LOSS)."""
        return [t for t in self._read_all_trades() if t["outcome"] is not None]

    def get_all_trades(self) -> list[dict[str, Any]]:
        """Return every trade, resolved or not."""
        return self._read_all_trades()

    # ------------------------------------------------------------------
    # Summary / Analytics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Compute a full performance summary.

        Returns
        -------
        dict with keys:
            total_trades : int
                Number of resolved trades.
            wins : int
            losses : int
            win_rate : float or None
                wins / total_trades. None if no trades.
            total_pnl : float
                Sum of all realised P&L.
            avg_pnl_per_trade : float or None
                total_pnl / total_trades.
            current_bankroll : float
                starting_bankroll + total_pnl.
            largest_win : float or None
                Best single-trade P&L.
            largest_loss : float or None
                Worst single-trade P&L (will be negative).
            open_trades : int
                Number of unresolved trades.
            total_exposure : float
                Sum of size_usd for open trades (money at risk).
            calibration : dict
                Output of ``compute_calibration`` — see calibration.py.
        """
        all_trades = self._read_all_trades()
        resolved = [t for t in all_trades if t["outcome"] is not None]
        open_trades = [t for t in all_trades if t["outcome"] is None]

        total = len(resolved)
        wins = sum(1 for t in resolved if t["outcome"] == "WIN")
        losses = total - wins

        pnls = [t["pnl"] for t in resolved]
        total_pnl = sum(pnls) if pnls else 0.0

        # Build calibration data from resolved trades
        # Each entry: (estimated_probability, did_the_event_happen)
        cal_data: list[tuple[float, bool]] = []
        for t in resolved:
            p_est = t["p_estimated"]
            # For YES bets, a WIN means the event happened -> True
            # For NO bets, a WIN means the event did NOT happen -> False
            if t["side"] == "YES":
                event_happened = t["outcome"] == "WIN"
            else:
                event_happened = t["outcome"] == "LOSS"
            cal_data.append((p_est, event_happened))

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total, 4) if total > 0 else None,
            "total_pnl": round(total_pnl, 4),
            "avg_pnl_per_trade": round(total_pnl / total, 4) if total > 0 else None,
            "current_bankroll": round(self.starting_bankroll + total_pnl, 4),
            "largest_win": round(max(pnls), 4) if pnls else None,
            "largest_loss": round(min(pnls), 4) if pnls else None,
            "open_trades": len(open_trades),
            "total_exposure": round(sum(t["size_usd"] for t in open_trades), 4),
            "calibration": compute_calibration(cal_data) if cal_data else {},
        }

    def print_summary(self) -> None:
        """Pretty-print the performance summary to the console."""
        s = self.get_summary()

        print("\n" + "=" * 50)
        print("  TRADE LOG SUMMARY")
        print("=" * 50)

        print(f"\n  Starting bankroll:   ${self.starting_bankroll:,.2f}")
        print(f"  Current bankroll:    ${s['current_bankroll']:,.2f}")
        pnl = s["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        print(f"  Total P&L:           {pnl_sign}${pnl:,.2f}")

        print(f"\n  Resolved trades:     {s['total_trades']}")
        print(f"  Wins:                {s['wins']}")
        print(f"  Losses:              {s['losses']}")
        if s["win_rate"] is not None:
            print(f"  Win rate:            {s['win_rate']:.1%}")
        if s["avg_pnl_per_trade"] is not None:
            avg = s["avg_pnl_per_trade"]
            avg_sign = "+" if avg >= 0 else ""
            print(f"  Avg P&L per trade:   {avg_sign}${avg:,.2f}")

        if s["largest_win"] is not None:
            print(f"\n  Largest win:         +${s['largest_win']:,.2f}")
        if s["largest_loss"] is not None:
            print(f"  Largest loss:        -${abs(s['largest_loss']):,.2f}")

        print(f"\n  Open trades:         {s['open_trades']}")
        print(f"  Capital at risk:     ${s['total_exposure']:,.2f}")

        # Calibration
        if s["calibration"]:
            all_trades = self._read_all_trades()
            resolved = [t for t in all_trades if t["outcome"] is not None]
            cal_data: list[tuple[float, bool]] = []
            for t in resolved:
                p_est = t["p_estimated"]
                if t["side"] == "YES":
                    event_happened = t["outcome"] == "WIN"
                else:
                    event_happened = t["outcome"] == "LOSS"
                cal_data.append((p_est, event_happened))
            print_calibration(cal_data)
        else:
            print("\n  (No calibration data yet — resolve some trades first)")

        print("=" * 50 + "\n")
