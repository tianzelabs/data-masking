"""Pure-Python aggregate statistics, backend-agnostic (operates on plain
lists produced by dataframe_adapter.column_values / sample rows).

Deliberately favors binned/quantile summaries over raw min/max for
sensitive columns: a raw min or max IS a real individual record's value
(the single highest salary, the single most extreme age), so it leaks
exactly like any other cell value would.
"""
from __future__ import annotations

import math
from collections import Counter


def _nice_step(span: float) -> float:
    """Round a bin width up to 1/2/5 * 10^k, so bin edges land on round
    numbers instead of exact order statistics of the data."""
    if span <= 0:
        return 1.0
    exp = math.floor(math.log10(span))
    base = span / (10**exp)
    nice = 1 if base <= 1 else 2 if base <= 2 else 5 if base <= 5 else 10
    return nice * (10**exp)


def numeric_quantile_bins(values: list[float], bins: int = 10, min_count: int = 5) -> list[dict]:
    """Equal-width histogram with two independent safeguards against a bin
    boundary ever equaling one real record's raw value:

    1. A k-anonymity floor -- every emitted bin covers at least `min_count`
       records (undersized bins are merged into a neighbor). Returns []
       if there are fewer than `min_count` values total: with that few
       records, no bin can be built without one containing a single
       value. The caller should treat an empty result as "suppressed",
       not "no data" (see summarize.py).
    2. Bin edges are rounded outward to a coarse 1/2/5x10^k grid
       (_nice_step) rather than taken from the data's own min/max or
       quantile order statistics, so an edge coincides with a genuine
       record's exact value only by rare coincidence, not systematically
       (as plain quantile-cut edges would, every time, at the two
       outermost bins).

    This is a pragmatic mitigation, not a formal (e.g. differential
    privacy) guarantee -- for a hard regulatory requirement, replace the
    rounding with fixed, dataset-independent bucket boundaries you define
    up front (e.g. business-meaningful salary bands) instead of ones
    derived from this column's own min/max.
    """
    vals = sorted(float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    n = len(vals)
    if n < min_count:
        return []

    lo, hi = vals[0], vals[-1]
    if lo == hi:
        pad = _nice_step(abs(lo) if lo != 0 else 1.0)
        return [{"range": [lo - pad, hi + pad], "count": n}]

    target_bins = max(1, min(bins, n // min_count))
    step = _nice_step((hi - lo) / target_bins)
    nice_lo = math.floor(lo / step) * step
    nice_hi = math.ceil(hi / step) * step
    n_bins = max(1, round((nice_hi - nice_lo) / step))
    edges = [nice_lo + i * step for i in range(n_bins + 1)]

    counts = [0] * n_bins
    for v in vals:
        idx = min(n_bins - 1, int((v - nice_lo) / step))
        counts[idx] += 1

    out: list[dict] = []
    i = 0
    while i < n_bins:
        lo_e, hi_e, c = edges[i], edges[i + 1], counts[i]
        j = i
        while c < min_count and j + 1 < n_bins:
            j += 1
            hi_e = edges[j + 1]
            c += counts[j]
        if c > 0:
            out.append({"range": [lo_e, hi_e], "count": c})
        i = j + 1
    if len(out) >= 2 and out[-1]["count"] < min_count:
        last = out.pop()
        out[-1]["range"][1] = last["range"][1]
        out[-1]["count"] += last["count"]
    return out


def numeric_full_stats(values: list[float]) -> dict:
    """Only call this for columns explicitly marked NOT sensitive -- unlike
    numeric_quantile_bins, min/max here are real individual values."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"count": 0}
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    svals = sorted(vals)
    return {
        "count": n,
        "min": svals[0],
        "max": svals[-1],
        "mean": mean,
        "std": math.sqrt(variance),
        "p25": svals[int(0.25 * (n - 1))],
        "median": svals[int(0.5 * (n - 1))],
        "p75": svals[int(0.75 * (n - 1))],
    }


def value_counts(values: list, limit: int = 50) -> dict[str, int] | None:
    """Only call this for columns explicitly marked NOT sensitive. Returns
    None if cardinality exceeds `limit` (i.e. this isn't a small enum)."""
    counts = Counter(str(v) for v in values)
    if len(counts) > limit:
        return None
    return dict(counts.most_common())
