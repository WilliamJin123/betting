"""
Entry Timing Coordinator
=========================

Higher-level module that ties together the OBI signals with the Polymarket
client. Instead of manually fetching order books and calling compute_obi(),
you use EntryTimer which handles the plumbing.

Two modes:
    1. check_entry() — one-shot check. Fetch the order book, compute signals,
       return immediately. Use this when you just want to display the signal.

    2. wait_for_entry() — blocking poll. Repeatedly check the order book
       until conditions are favorable (or max wait is exceeded). Use this
       when you actually want to delay order placement until the price dips.

Usage:
    from bot.signals.timing import EntryTimer

    timer = EntryTimer(client)

    # Quick check (non-blocking)
    signal = timer.check_entry(token_id="abc123")
    print(signal["reason"])

    # Wait for good entry (blocking, up to 30 min)
    signal = timer.wait_for_entry(token_id="abc123", max_wait_minutes=30)
    if signal["signal"] in ("buy_now", "strong_buy"):
        place_order(...)
"""

import logging
import time

from bot.polymarket.client import PolymarketClient
from bot.signals.obi import should_wait_for_entry

logger = logging.getLogger(__name__)


class EntryTimer:
    """
    Monitors a market's order book and suggests optimal entry timing.

    This is a thin wrapper around the OBI functions that handles fetching
    the order book from the API. It doesn't place orders -- it just tells
    you WHEN to place them.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client

    def check_entry(self, token_id: str) -> dict:
        """
        Fetch the current order book and return an entry signal.

        This is a single non-blocking check. It makes one API call to get
        the order book, computes the OBI metrics, and returns the signal.

        Args:
            token_id: The outcome token's ID (from market.yes_token_id etc).

        Returns:
            dict with signal, obi, imbalance_ratio, vamp, midpoint, reason,
            and suggested_wait_minutes. See obi.should_wait_for_entry() for
            full details.

        Raises:
            requests.HTTPError: If the API call fails.
        """
        order_book = self.client.get_order_book(token_id)
        return should_wait_for_entry(order_book, side="BUY")

    def wait_for_entry(
        self,
        token_id: str,
        max_wait_minutes: int = 30,
        check_interval_seconds: int = 60,
    ) -> dict:
        """
        Poll the order book until conditions are favorable or max_wait is reached.

        This is a BLOCKING call. It will sleep between checks. Use it when
        you want the bot to automatically wait for a price dip before entering.

        Logic:
            1. Check the order book.
            2. If signal is "buy_now" or "strong_buy" -- return immediately.
            3. If signal is "wait" -- sleep for check_interval_seconds, then
               re-check.
            4. If we've waited longer than max_wait_minutes -- return the
               latest signal regardless (the caller decides what to do).

        Args:
            token_id:               The outcome token's ID.
            max_wait_minutes:       Maximum minutes to wait before giving up.
            check_interval_seconds: Seconds between order book checks.

        Returns:
            The final signal dict (same format as check_entry). The signal
            field will be "buy_now", "strong_buy", or "wait" (if we timed out).
            An extra key "waited_seconds" is added showing how long we waited.
        """
        start_time = time.monotonic()
        max_wait_seconds = max_wait_minutes * 60
        checks = 0

        while True:
            checks += 1
            elapsed = time.monotonic() - start_time

            signal = self.check_entry(token_id)
            signal["waited_seconds"] = int(elapsed)
            signal["checks_performed"] = checks

            logger.info(
                "Entry check #%d (%.0fs elapsed): signal=%s, obi=%.3f, ir=%.3f",
                checks, elapsed, signal["signal"], signal["obi"],
                signal["imbalance_ratio"],
            )

            # Good to go
            if signal["signal"] in ("buy_now", "strong_buy"):
                return signal

            # Timed out
            if elapsed >= max_wait_seconds:
                logger.info(
                    "Max wait of %d minutes reached. Returning current signal: %s",
                    max_wait_minutes, signal["signal"],
                )
                signal["reason"] = (
                    f"Waited {max_wait_minutes}min but conditions didn't improve. "
                    f"Last OBI={signal['obi']:+.2f}. Proceeding anyway."
                )
                return signal

            # Wait and try again
            remaining = max_wait_seconds - elapsed
            sleep_time = min(check_interval_seconds, remaining)
            if sleep_time > 0:
                logger.debug("Sleeping %.0fs before next check...", sleep_time)
                time.sleep(sleep_time)
