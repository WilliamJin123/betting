"""
Calibration Checker — Are our probability estimates actually accurate?

WHY THIS MATTERS:
    When our bot says "I think there's a 70% chance this happens," it should
    actually happen about 70% of the time. If we say 70% but it only happens
    40% of the time, our model is badly miscalibrated and we'll lose money.

    Calibration is checked by:
    1. Grouping all our predictions into "buckets" (0-10%, 10-20%, ..., 90-100%)
    2. Within each bucket, computing the actual win rate
    3. Comparing: if the average prediction in the 70-80% bucket is 75%,
       the actual win rate should be close to 75%

    A perfectly calibrated model has predicted probability == actual frequency
    for every bucket. In practice, you want them to be close.

HOW TO READ THE OUTPUT:
    Bucket [0.60, 0.70):  avg_predicted=0.65  actual_rate=0.63  n=15
    This means: for 15 bets where we predicted 60-70%, the events actually
    happened 63% of the time. Our predictions of ~65% were close. Good!

    If actual_rate is way higher than avg_predicted, we're underconfident.
    If actual_rate is way lower than avg_predicted, we're overconfident.
"""

from __future__ import annotations

from typing import Sequence


# ---------------------------------------------------------------------------
# Bucket boundaries — ten bins from 0% to 100%
# ---------------------------------------------------------------------------
BUCKET_EDGES: list[float] = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0


def _bucket_label(lo: float, hi: float) -> str:
    """Human-readable label for a bucket, e.g. '[0.60, 0.70)'."""
    return f"[{lo:.2f}, {hi:.2f})"


def compute_calibration(
    predictions: Sequence[tuple[float, bool]],
) -> dict[str, dict]:
    """Compute calibration statistics for a list of (predicted_prob, actual_outcome).

    Parameters
    ----------
    predictions : list of (float, bool)
        Each element is (estimated_probability, did_event_happen).
        - estimated_probability is between 0 and 1
        - did_event_happen is True if the event occurred, False otherwise

    Returns
    -------
    dict
        Keyed by bucket label string.  Each value is a dict with:
            - ``bucket_lo``: lower bound (inclusive)
            - ``bucket_hi``: upper bound (exclusive, except for the last bucket)
            - ``count``: number of predictions in this bucket
            - ``avg_predicted``: mean predicted probability in the bucket
            - ``actual_rate``: fraction that actually happened
            - ``gap``: actual_rate - avg_predicted (positive = underconfident)

    Example
    -------
    >>> data = [(0.75, True), (0.72, False), (0.80, True)]
    >>> cal = compute_calibration(data)
    >>> cal["[0.70, 0.80)"]["count"]
    2
    """
    # Initialise accumulators for each bucket
    buckets: dict[str, list[tuple[float, bool]]] = {}
    for i in range(len(BUCKET_EDGES) - 1):
        label = _bucket_label(BUCKET_EDGES[i], BUCKET_EDGES[i + 1])
        buckets[label] = []

    # Sort each prediction into the right bucket
    for prob, outcome in predictions:
        placed = False
        for i in range(len(BUCKET_EDGES) - 1):
            lo = BUCKET_EDGES[i]
            hi = BUCKET_EDGES[i + 1]
            # Last bucket is inclusive on both ends: [0.90, 1.00]
            if i == len(BUCKET_EDGES) - 2:
                in_bucket = lo <= prob <= hi
            else:
                in_bucket = lo <= prob < hi
            if in_bucket:
                label = _bucket_label(lo, hi)
                buckets[label].append((prob, outcome))
                placed = True
                break
        if not placed:
            # Edge case: prob exactly 1.0 should land in last bucket
            label = _bucket_label(BUCKET_EDGES[-2], BUCKET_EDGES[-1])
            buckets[label].append((prob, outcome))

    # Compute stats for each bucket
    result: dict[str, dict] = {}
    for i in range(len(BUCKET_EDGES) - 1):
        lo = BUCKET_EDGES[i]
        hi = BUCKET_EDGES[i + 1]
        label = _bucket_label(lo, hi)
        items = buckets[label]
        count = len(items)

        if count == 0:
            result[label] = {
                "bucket_lo": lo,
                "bucket_hi": hi,
                "count": 0,
                "avg_predicted": None,
                "actual_rate": None,
                "gap": None,
            }
        else:
            avg_pred = sum(p for p, _ in items) / count
            actual = sum(1 for _, o in items if o) / count
            result[label] = {
                "bucket_lo": lo,
                "bucket_hi": hi,
                "count": count,
                "avg_predicted": round(avg_pred, 4),
                "actual_rate": round(actual, 4),
                "gap": round(actual - avg_pred, 4),
            }

    return result


def print_calibration(
    predictions: Sequence[tuple[float, bool]],
) -> dict[str, dict]:
    """Compute calibration and pretty-print a table to the console.

    Returns the same dict as ``compute_calibration`` so you can use
    the data programmatically too.
    """
    cal = compute_calibration(predictions)

    print("\n=== Calibration Report ===")
    print(f"{'Bucket':<16} {'Count':>5}  {'Avg Pred':>9}  {'Actual':>7}  {'Gap':>7}")
    print("-" * 54)

    for label, stats in cal.items():
        count = stats["count"]
        if count == 0:
            print(f"{label:<16} {count:>5}  {'---':>9}  {'---':>7}  {'---':>7}")
        else:
            avg_p = stats["avg_predicted"]
            act = stats["actual_rate"]
            gap = stats["gap"]
            # Colour-code the gap: negative means overconfident
            gap_str = f"{gap:+.2%}"
            print(
                f"{label:<16} {count:>5}  {avg_p:>8.1%}  {act:>6.1%}  {gap_str:>7}"
            )

    print()
    return cal
