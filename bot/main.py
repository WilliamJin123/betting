"""
Polymarket Bot — Main Entry Point
==================================

This is the orchestrator that ties every module together. It exposes a CLI
with subcommands for scanning markets, placing bets, resolving outcomes,
and checking portfolio status.

Usage:
    python -m bot scan                          # Scan for opportunities
    python -m bot bet <market_id> <p_true>      # Place a bet with Kelly sizing
    python -m bot resolve                       # Check for resolved markets
    python -m bot status                        # Portfolio summary
    python -m bot paper-scan                    # Auto-bet on scan results (paper mode)
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import NoReturn

from bot import config
from datetime import datetime, timezone

from bot.polymarket.client import PolymarketClient
from bot.polymarket.models import Market
from bot.sizing.kelly import kelly_bet
from bot.tracker.trade_logger import TradeLogger
from bot.tracker.resolver import MarketResolver

logger = logging.getLogger("bot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the whole bot."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_client() -> PolymarketClient:
    """Create and health-check a PolymarketClient."""
    client = PolymarketClient()
    return client


def _make_logger() -> TradeLogger:
    """Create a TradeLogger with the configured bankroll."""
    return TradeLogger(starting_bankroll=config.INITIAL_BANKROLL)


def _truncate(text: str, width: int) -> str:
    """Truncate text to `width` characters, adding '...' if trimmed."""
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _days_to_resolution(end_date_str: str | None) -> float | None:
    """Parse a market's end_date and return days from now, or None."""
    if not end_date_str:
        return None
    try:
        cleaned = end_date_str.replace("Z", "+00:00")
        end_dt = datetime.fromisoformat(cleaned)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        delta = end_dt - datetime.now(timezone.utc)
        days = delta.total_seconds() / 86400.0
        return max(days, 0.0)
    except (ValueError, TypeError):
        return None


def _format_price(price: float | None) -> str:
    """Format a price for display."""
    if price is None:
        return "  --  "
    return f"{price:.2f}"


def _print_market_table(markets: list[Market], title: str = "Markets") -> None:
    """Print a formatted table of markets."""
    if not markets:
        print(f"\n  No markets to display.")
        return

    print(f"\n  {title}")
    print(f"  {'=' * 90}")
    print(
        f"  {'#':<4} {'Question':<45} {'YES':>6} {'NO':>6} "
        f"{'Spread':>7} {'Vol 24h':>10}"
    )
    print(f"  {'-' * 90}")

    for i, m in enumerate(markets, 1):
        question = _truncate(m.question, 43)
        yes_p = _format_price(m.yes_price)
        no_p = _format_price(m.no_price)
        spread = f"{m.spread:.3f}" if m.spread else "  --  "
        vol = f"${m.volume_24h:,.0f}" if m.volume_24h else "  --  "
        print(f"  {i:<4} {question:<45} {yes_p:>6} {no_p:>6} {spread:>7} {vol:>10}")

    print(f"  {'=' * 90}\n")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> None:
    """Scan for market opportunities and display them."""
    # Import scanner here so the module can be loaded even if scanner
    # isn't built yet (it's being built in parallel).
    from bot.scanner.market_scanner import MarketScanner

    print("\n  Scanning Polymarket for opportunities...")
    client = _make_client()
    scanner = MarketScanner(client)

    try:
        results = scanner.run_full_scan()
    except Exception as e:
        print(f"\n  Error during scan: {e}")
        return

    if not results:
        print("  No opportunities found. Markets may be efficient right now.")
        return

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)

    # Limit output
    limit = args.limit if hasattr(args, "limit") else 20
    results = results[:limit]

    print(f"\n  Found {len(results)} opportunities (showing top {limit}):")
    print(f"  {'=' * 100}")
    print(
        f"  {'#':<4} {'Type':<14} {'Score':>5} {'Question':<40} "
        f"{'YES':>6} {'NO':>6} {'Spread':>7} {'Vol 24h':>10}"
    )
    print(f"  {'-' * 100}")

    for i, result in enumerate(results, 1):
        m = result.market
        question = _truncate(m.question, 38)
        yes_p = _format_price(m.yes_price)
        no_p = _format_price(m.no_price)
        spread = f"{m.spread:.3f}" if m.spread else "  --  "
        vol = f"${m.volume_24h:,.0f}" if m.volume_24h else "  --  "
        print(
            f"  {i:<4} {result.opportunity_type:<14} {result.score:>5.2f} "
            f"{question:<40} {yes_p:>6} {no_p:>6} {spread:>7} {vol:>10}"
        )

        # Show key details for each opportunity
        if result.details:
            detail_parts = []
            for k, v in result.details.items():
                if isinstance(v, float):
                    detail_parts.append(f"{k}={v:.4f}")
                else:
                    detail_parts.append(f"{k}={v}")
            detail_str = ", ".join(detail_parts)
            print(f"       -> {detail_str}")

    print(f"  {'=' * 100}\n")


def cmd_bet(args: argparse.Namespace) -> None:
    """Place a bet on a specific market using Kelly sizing."""
    from bot.polymarket.executor import OrderExecutor

    market_id: str = args.market_id
    p_true: float = args.p_true

    if not (0 < p_true < 1):
        print(f"\n  Error: p_true must be between 0 and 1 (exclusive), got {p_true}")
        return

    client = _make_client()
    trade_logger = _make_logger()
    live_mode: bool = getattr(args, "live", False)
    executor = OrderExecutor(mode="live" if live_mode else "paper")

    # Fetch market data
    print(f"\n  Fetching market {market_id}...")
    market = client.get_clob_market(market_id)
    if market is None:
        # Try Gamma API with market_id as a numeric ID
        market = client.get_gamma_market_by_id(market_id)
    if market is None:
        print(f"  Error: Market '{market_id}' not found.")
        return

    print(f"  Market: {market.question}")
    print(f"  YES price: {_format_price(market.yes_price)}")
    print(f"  NO price:  {_format_price(market.no_price)}")

    # Determine market price for Kelly
    # For YES side: p_market = yes_price
    # For NO side: p_market = yes_price (Kelly figures out direction)
    p_market = market.yes_price
    if p_market is None or p_market <= 0 or p_market >= 1:
        print(f"  Error: Invalid market price ({p_market}). Cannot size bet.")
        return

    # Forced side override
    forced_side = getattr(args, "side", None)

    # Run Kelly sizing
    summary = trade_logger.get_summary()
    bankroll = summary["current_bankroll"]
    days = _days_to_resolution(market.end_date)

    bet_result = kelly_bet(
        p_true=p_true,
        p_market=p_market,
        bankroll=bankroll,
        kelly_fraction_multiplier=config.KELLY_MULTIPLIER,
        max_bet_fraction=config.MAX_BET_FRACTION,
        min_edge=config.MIN_EDGE,
        days_to_resolution=days,
    )

    # If user forced a side and Kelly disagrees, warn them
    if forced_side and forced_side != bet_result["side"]:
        print(
            f"\n  Warning: You requested {forced_side} but Kelly suggests "
            f"{bet_result['side']} (edge={bet_result['edge']:.4f})."
        )
        print(f"  Overriding Kelly side to {forced_side}.")
        bet_result["side"] = forced_side

    print(f"\n  Kelly Analysis:")
    print(f"    Your estimate:  {p_true:.1%}")
    print(f"    Market price:   {p_market:.1%}")
    print(f"    Edge:           {bet_result['abs_edge']:.1%}")
    print(f"    Raw Kelly:      {bet_result['kelly_f']:.1%}")
    print(f"    Adjusted frac:  {bet_result['adjusted_f']:.1%}")
    print(f"    Bankroll:       ${bankroll:.2f}")
    print(f"    Bet size:       ${bet_result['size']:.2f}")
    print(f"    Side:           {bet_result['side']}")
    print(f"    Reason:         {bet_result['reason']}")

    if bet_result["size"] <= 0:
        print(f"\n  No bet placed: insufficient edge.")
        return

    # Determine entry price and token
    side = bet_result["side"]
    if side == "YES":
        entry_price = market.yes_price
        token_id = market.yes_token_id
    else:
        entry_price = market.no_price
        token_id = market.no_token_id

    if entry_price is None or entry_price <= 0 or entry_price >= 1:
        print(f"  Error: Invalid entry price for {side} side ({entry_price}).")
        return

    if token_id is None:
        print(f"  Error: No token ID found for {side} side.")
        return

    # Place order
    mode_str = "LIVE" if live_mode else "PAPER"
    print(f"\n  Placing {mode_str} order: {side} ${bet_result['size']:.2f} @ {entry_price:.4f}...")

    try:
        order_result = executor.place_order(
            token_id=token_id,
            side="BUY",
            size_usd=bet_result["size"],
            price=entry_price,
        )
        print(f"  Order result: {order_result}")
    except Exception as e:
        print(f"  Error placing order: {e}")
        return

    # Log the trade
    trade_record = trade_logger.log_trade(
        market_id=market.condition_id,
        market_question=market.question,
        side=side,
        size_usd=bet_result["size"],
        entry_price=entry_price,
        p_estimated=p_true,
        p_market=p_market,
        edge=bet_result["edge"],
        kelly_fraction=bet_result["adjusted_f"],
    )

    print(f"\n  Trade logged successfully.")
    print(f"    Market:     {trade_record['market_question']}")
    print(f"    Side:       {trade_record['side']}")
    print(f"    Size:       ${trade_record['size_usd']:.2f}")
    print(f"    Entry:      {trade_record['entry_price']:.4f}")
    print(f"    Shares:     {trade_record['shares']:.4f}")
    print()


def cmd_resolve(args: argparse.Namespace) -> None:
    """Check for resolved markets and update P&L."""
    client = _make_client()
    trade_logger = _make_logger()
    resolver = MarketResolver(client, trade_logger)

    print("\n  Checking open trades for resolution...")

    newly_resolved = resolver.check_resolutions()

    if not newly_resolved:
        open_count = len(trade_logger.get_open_trades())
        print(f"  No new resolutions. {open_count} trades still open.")
    else:
        print(f"\n  Newly resolved trades:")
        print(f"  {'-' * 70}")
        for trade in newly_resolved:
            pnl = trade.get("pnl", 0)
            pnl_sign = "+" if pnl >= 0 else ""
            outcome = trade.get("outcome", "?")
            print(
                f"  {outcome:<5} | {pnl_sign}${pnl:.2f} | "
                f"{trade['side']} @ {trade['entry_price']:.4f} | "
                f"{_truncate(trade['market_question'], 40)}"
            )
        print(f"  {'-' * 70}")

        total_pnl = sum(t.get("pnl", 0) for t in newly_resolved)
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"  Net P&L from resolutions: {pnl_sign}${total_pnl:.2f}")

    print()


def cmd_status(args: argparse.Namespace) -> None:
    """Print full portfolio status and P&L summary."""
    trade_logger = _make_logger()

    all_trades = trade_logger.get_all_trades()
    if not all_trades:
        print("\n  No trades recorded yet. Use 'scan' to find opportunities,")
        print("  then 'bet' or 'paper-scan' to place trades.")
        print()
        return

    # Print the built-in summary (handles bankroll, P&L, calibration)
    trade_logger.print_summary()

    # Also show open trades in detail
    open_trades = trade_logger.get_open_trades()
    if open_trades:
        print(f"  Open Trades ({len(open_trades)}):")
        print(f"  {'-' * 80}")
        print(
            f"  {'Side':<5} {'Size':>8} {'Entry':>7} {'P_est':>7} "
            f"{'Edge':>7} {'Market Question':<40}"
        )
        print(f"  {'-' * 80}")
        for t in open_trades:
            question = _truncate(t["market_question"], 38)
            print(
                f"  {t['side']:<5} ${t['size_usd']:>7.2f} "
                f"{t['entry_price']:>7.4f} {t['p_estimated']:>7.2%} "
                f"{t['edge']:>+7.2%} {question:<40}"
            )
        print(f"  {'-' * 80}\n")

    # Show recent resolved trades
    resolved = trade_logger.get_resolved_trades()
    if resolved:
        recent = resolved[-10:]  # last 10
        print(f"  Recent Resolved Trades (last {len(recent)} of {len(resolved)}):")
        print(f"  {'-' * 85}")
        print(
            f"  {'Result':<6} {'P&L':>9} {'Side':<5} {'Entry':>7} "
            f"{'P_est':>7} {'Market Question':<40}"
        )
        print(f"  {'-' * 85}")
        for t in recent:
            question = _truncate(t["market_question"], 38)
            pnl = t["pnl"]
            pnl_str = f"{'+'if pnl >= 0 else ''}${pnl:.2f}"
            print(
                f"  {t['outcome']:<6} {pnl_str:>9} {t['side']:<5} "
                f"{t['entry_price']:>7.4f} {t['p_estimated']:>7.2%} "
                f"{question:<40}"
            )
        print(f"  {'-' * 85}\n")


def cmd_paper_scan(args: argparse.Namespace) -> None:
    """Scan markets and auto-bet in paper mode for pipeline testing."""
    from bot.scanner.market_scanner import MarketScanner
    from bot.polymarket.executor import OrderExecutor

    client = _make_client()
    trade_logger = _make_logger()
    scanner = MarketScanner(client)
    executor = OrderExecutor(mode="paper")

    print("\n  [Paper Scan] Running full pipeline in paper mode...")
    print(f"  Bankroll: ${trade_logger.get_summary()['current_bankroll']:.2f}")
    print(f"  Kelly multiplier: {config.KELLY_MULTIPLIER}")
    print(f"  Min edge: {config.MIN_EDGE:.0%}")
    print(f"  Max bet fraction: {config.MAX_BET_FRACTION:.0%}")

    # Step 1: Scan
    print("\n  Step 1: Scanning for opportunities...")
    try:
        results = scanner.run_full_scan()
    except Exception as e:
        print(f"  Error during scan: {e}")
        return

    if not results:
        print("  No opportunities found.")
        return

    results.sort(key=lambda r: r.score, reverse=True)
    score_threshold = getattr(args, "threshold", 0.3)
    filtered = [r for r in results if r.score >= score_threshold]

    print(f"  Found {len(results)} total opportunities, {len(filtered)} above threshold ({score_threshold:.1f}).")

    if not filtered:
        print("  No opportunities above threshold. Try lowering --threshold.")
        return

    # Limit how many we auto-bet on
    max_bets = getattr(args, "max_bets", 5)
    filtered = filtered[:max_bets]

    # Step 2: Size and place paper bets
    print(f"\n  Step 2: Sizing and placing up to {len(filtered)} paper bets...\n")

    bets_placed = 0
    bets_skipped = 0
    summary = trade_logger.get_summary()
    bankroll = summary["current_bankroll"]

    for i, result in enumerate(filtered, 1):
        market = result.market
        print(f"  [{i}] {_truncate(market.question, 60)}")
        print(f"      Type: {result.opportunity_type}, Score: {result.score:.3f}")

        p_market = market.yes_price
        if p_market is None or p_market <= 0 or p_market >= 1:
            print(f"      Skipped: invalid market price ({p_market})")
            bets_skipped += 1
            continue

        # For paper scan, we use a simple heuristic for p_true:
        # We look at what the scanner found and nudge the probability
        # in the direction the opportunity suggests.
        p_true = _estimate_p_true_from_scan(result, p_market)
        if p_true is None:
            print(f"      Skipped: could not estimate p_true")
            bets_skipped += 1
            continue

        days = _days_to_resolution(market.end_date)
        try:
            bet_result = kelly_bet(
                p_true=p_true,
                p_market=p_market,
                bankroll=bankroll,
                kelly_fraction_multiplier=config.KELLY_MULTIPLIER,
                max_bet_fraction=config.MAX_BET_FRACTION,
                min_edge=config.MIN_EDGE,
                days_to_resolution=days,
            )
        except (ValueError, TypeError) as e:
            print(f"      Skipped: Kelly error: {e}")
            bets_skipped += 1
            continue

        if bet_result["size"] <= 0:
            print(f"      Skipped: {bet_result['reason']}")
            bets_skipped += 1
            continue

        side = bet_result["side"]
        if side == "YES":
            entry_price = market.yes_price
            token_id = market.yes_token_id
        else:
            entry_price = market.no_price
            token_id = market.no_token_id

        if entry_price is None or entry_price <= 0 or entry_price >= 1:
            print(f"      Skipped: invalid entry price for {side}")
            bets_skipped += 1
            continue

        if token_id is None:
            print(f"      Skipped: no token ID for {side}")
            bets_skipped += 1
            continue

        # Place paper order
        # Kelly returns side as "YES"/"NO" (which token to bet on),
        # but the executor expects "BUY"/"SELL" (order direction).
        # Buying a YES or NO token is always a "BUY" order.
        try:
            executor.place_order(
                token_id=token_id,
                side="BUY",
                size_usd=bet_result["size"],
                price=entry_price,
            )
        except Exception as e:
            print(f"      Error placing paper order: {e}")
            bets_skipped += 1
            continue

        # Log the trade
        trade_logger.log_trade(
            market_id=market.condition_id,
            market_question=market.question,
            side=side,
            size_usd=bet_result["size"],
            entry_price=entry_price,
            p_estimated=p_true,
            p_market=p_market,
            edge=bet_result["edge"],
            kelly_fraction=bet_result["adjusted_f"],
        )

        print(
            f"      -> {side} ${bet_result['size']:.2f} @ {entry_price:.4f} "
            f"(edge={bet_result['abs_edge']:.1%}, kelly={bet_result['adjusted_f']:.1%})"
        )
        bets_placed += 1

        # Update bankroll for next bet (reduce available capital)
        bankroll -= bet_result["size"]
        if bankroll <= 0:
            print("\n  Bankroll exhausted. Stopping.")
            break

    # Step 3: Summary
    print(f"\n  {'=' * 50}")
    print(f"  Paper Scan Complete")
    print(f"  {'=' * 50}")
    print(f"  Bets placed:  {bets_placed}")
    print(f"  Bets skipped: {bets_skipped}")
    print(f"  Remaining bankroll: ${bankroll:.2f}")
    print()

    if bets_placed > 0:
        print("  Use 'python -m bot status' to see your paper portfolio.")
        print("  Use 'python -m bot resolve' to check for resolutions.\n")


def _estimate_p_true_from_scan(result, p_market: float) -> float | None:
    """Derive a p_true estimate from scanner results.

    In paper-scan mode, we need a probability estimate to feed to Kelly.
    The scanner flags opportunities but doesn't give us a probability.

    Strategy:
        - For "arb" opportunities: the price gap IS the edge. If prices
          sum to 0.94, there's ~6% mispricing. We shift p_true by half
          the gap in the favorable direction.
        - For "wide_spread" opportunities: the midpoint is likely closer
          to fair value. We use a small edge from the midpoint.
        - For "stale" opportunities: we assume the stale price is ~5-10%
          off from fair value and nudge toward 0.50.
        - For anything else: nudge by a small default edge.

    Returns None if we can't form a reasonable estimate.
    """
    details = result.details
    opp_type = result.opportunity_type

    # Default edge to assume when we don't have better information
    DEFAULT_EDGE = 0.07  # 7 percentage points

    if opp_type == "arb":
        gap = details.get("gap", details.get("profit_per_dollar", 0))
        if gap and gap > 0:
            # Shift p_true by half the arb gap
            shift = min(gap / 2, 0.15)
            p_true = p_market + shift
        else:
            p_true = p_market + DEFAULT_EDGE

    elif opp_type == "wide_spread":
        # Use half the spread as our estimated edge
        spread = details.get("spread", 0)
        if spread > 0:
            shift = min(spread / 2, 0.15)
            # Bet toward 0.50 (mean reversion heuristic)
            if p_market < 0.5:
                p_true = p_market + shift
            else:
                p_true = p_market - shift
        else:
            p_true = p_market + DEFAULT_EDGE

    elif opp_type == "stale":
        # Stale markets are likely mispriced. Nudge toward 0.50.
        if p_market < 0.5:
            p_true = min(p_market + DEFAULT_EDGE, 0.95)
        else:
            p_true = max(p_market - DEFAULT_EDGE, 0.05)

    else:
        # Generic: assume a modest edge
        if p_market < 0.5:
            p_true = min(p_market + DEFAULT_EDGE, 0.95)
        else:
            p_true = max(p_market - DEFAULT_EDGE, 0.05)

    # Clamp to valid range
    p_true = max(0.01, min(0.99, p_true))

    # Sanity check: don't produce a p_true that's basically the same as p_market
    if abs(p_true - p_market) < 0.01:
        return None

    return p_true


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="bot",
        description="Polymarket prediction market trading bot.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- scan ---
    scan_parser = subparsers.add_parser(
        "scan", help="Scan markets for opportunities."
    )
    scan_parser.add_argument(
        "--limit", type=int, default=20,
        help="Max opportunities to display (default: 20).",
    )

    # --- bet ---
    bet_parser = subparsers.add_parser(
        "bet", help="Place a bet on a market with Kelly sizing."
    )
    bet_parser.add_argument(
        "market_id", type=str,
        help="Market condition_id or Gamma numeric ID.",
    )
    bet_parser.add_argument(
        "p_true", type=float,
        help="Your estimated true probability (0-1, exclusive).",
    )
    bet_parser.add_argument(
        "--side", type=str, choices=["YES", "NO"], default=None,
        help="Force a side (YES or NO). If omitted, Kelly decides.",
    )
    bet_parser.add_argument(
        "--live", action="store_true",
        help="Place a real order (default is paper mode).",
    )

    # --- resolve ---
    subparsers.add_parser(
        "resolve", help="Check for resolved markets and update P&L."
    )

    # --- status ---
    subparsers.add_parser(
        "status", help="Print portfolio status and P&L summary."
    )

    # --- paper-scan ---
    paper_parser = subparsers.add_parser(
        "paper-scan", help="Scan + auto-bet in paper mode (testing)."
    )
    paper_parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Min opportunity score to auto-bet on (default: 0.3).",
    )
    paper_parser.add_argument(
        "--max-bets", type=int, default=5,
        help="Max paper bets to place per run (default: 5).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the bot CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    _setup_logging(verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "scan": cmd_scan,
        "bet": cmd_bet,
        "resolve": cmd_resolve,
        "status": cmd_status,
        "paper-scan": cmd_paper_scan,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unhandled error in '%s' command", args.command)
        print(f"\n  Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
