"""
Backtest Results
=================

Dataclass and reporting for backtesting output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    """
    Full result of a backtesting run.

    Fields:
        total_markets:    Markets the strategy had an opinion on.
        bets_placed:      Bets that passed Kelly's min_edge filter.
        bets_skipped:     Bets below the minimum edge threshold.
        wins:             Number of winning bets.
        losses:           Number of losing bets.
        win_rate:         wins / bets_placed (0 if no bets).
        total_pnl:        Total profit/loss in USD.
        final_bankroll:   Ending bankroll after all bets.
        max_drawdown:     Worst peak-to-trough decline as a fraction (0-1).
        sharpe_ratio:     Approximate annualized Sharpe ratio.
        avg_edge:         Average absolute edge across all placed bets.
        avg_bet_size:     Average bet size in USD.
        calibration:      Calibration dict from compute_calibration().
        trades:           List of individual trade record dicts.
    """
    total_markets: int = 0
    bets_placed: int = 0
    bets_skipped: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    final_bankroll: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_edge: float = 0.0
    avg_bet_size: float = 0.0
    calibration: dict = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)

    def print_report(self) -> None:
        """Print a formatted backtest report to the console."""
        print()
        print("=" * 60)
        print("  BACKTEST REPORT")
        print("=" * 60)
        print()

        # --- Overview ---
        print(f"  Markets evaluated:   {self.total_markets}")
        print(f"  Bets placed:         {self.bets_placed}")
        print(f"  Bets skipped:        {self.bets_skipped}  (below min edge)")
        print()

        # --- Win/Loss ---
        if self.bets_placed > 0:
            print(f"  Wins:                {self.wins}")
            print(f"  Losses:              {self.losses}")
            print(f"  Win rate:            {self.win_rate:.1%}")
        else:
            print("  No bets were placed (strategy never exceeded min edge).")
        print()

        # --- P&L ---
        pnl_sign = "+" if self.total_pnl >= 0 else ""
        print(f"  Total P&L:           {pnl_sign}${self.total_pnl:.4f}")
        print(f"  Final bankroll:      ${self.final_bankroll:.4f}")
        print(f"  Return:              {pnl_sign}{self._return_pct():.2%}")
        print()

        # --- Risk ---
        print(f"  Max drawdown:        {self.max_drawdown:.2%}")
        print(f"  Sharpe ratio:        {self.sharpe_ratio:.2f}  (approx. annualized)")
        print()

        # --- Sizing ---
        if self.bets_placed > 0:
            print(f"  Avg edge:            {self.avg_edge:.2%}")
            print(f"  Avg bet size:        ${self.avg_bet_size:.4f}")
            print()

        # --- Calibration ---
        if self.calibration:
            print("  --- Calibration ---")
            print(f"  {'Bucket':<16} {'Count':>5}  {'Avg Pred':>9}  {'Actual':>7}  {'Gap':>7}")
            print(f"  {'-' * 54}")
            for label, stats in self.calibration.items():
                count = stats["count"]
                if count == 0:
                    continue  # skip empty buckets for readability
                avg_p = stats["avg_predicted"]
                act = stats["actual_rate"]
                gap = stats["gap"]
                gap_str = f"{gap:+.2%}"
                print(
                    f"  {label:<16} {count:>5}  {avg_p:>8.1%}  {act:>6.1%}  {gap_str:>7}"
                )
            print()

        # --- Sample trades ---
        if self.trades:
            n_show = min(10, len(self.trades))
            print(f"  --- Sample Trades (first {n_show} of {len(self.trades)}) ---")
            print(f"  {'Result':<6} {'P&L':>9} {'Side':<4} {'P_est':>7} {'P_mkt':>7} {'Edge':>7}  {'Question'}")
            print(f"  {'-' * 80}")
            for t in self.trades[:n_show]:
                outcome = "WIN" if t["won"] else "LOSS"
                pnl = t["pnl"]
                pnl_str = f"{'+'if pnl >= 0 else ''}${pnl:.4f}"
                q = t["question"][:40]
                print(
                    f"  {outcome:<6} {pnl_str:>9} {t['side']:<4} "
                    f"{t['p_estimated']:>7.2%} {t['p_market']:>7.2%} "
                    f"{t['edge']:>+7.2%}  {q}"
                )
            print()

        print("=" * 60)
        print()

    def _return_pct(self) -> float:
        """Compute return as a percentage of the starting bankroll."""
        starting = self.final_bankroll - self.total_pnl
        if starting <= 0:
            return 0.0
        return self.total_pnl / starting
