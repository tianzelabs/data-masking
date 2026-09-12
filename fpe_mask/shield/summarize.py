"""summarize_for_agent(): the single entry point a marimo<->agent bridge
should call on a cell's result before it becomes the agent's tool response.

Design recap (see README.md "marimo / agent-shield" section):
- DataFrame -> schema + (binned, not raw) stats for sensitive columns, full
  stats/categories only for columns a policy explicitly marks non-sensitive,
  plus a format-preserving-masked sample of head() rows.
- scalar/str/list/dict/other -> a conservative, type-appropriate summary.
Nothing here ever forwards a raw sensitive value; pair it with
redact.redact_exception() for anything that goes through an except block.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import math
from typing import Any

from ..engine import MaskingEngine
from ..key_store import load_or_create_master_key
from .dataframe_adapter import ColumnInfo, column_values, is_dataframe
from .dataframe_adapter import adapt as _adapt_frame
from .policy import ColumnPolicy
from .redact import redact_text
from .registry import SensitiveValueRegistry
from .stats import numeric_full_stats, numeric_quantile_bins, value_counts

LOW_CARDINALITY_LIMIT = 50
DEFAULT_NUMERIC_BINS = 10
DEFAULT_SAMPLE_ROWS = 5

_engine_cache: dict[bytes, MaskingEngine] = {}


def _default_engine() -> MaskingEngine:
    key = load_or_create_master_key()
    eng = _engine_cache.get(key)
    if eng is None:
        eng = MaskingEngine(key, mode="standard")
        _engine_cache[key] = eng
    return eng


def _perturb_date(master_key: bytes, value, context: str, max_days: int = 30):
    """Deterministic +/-max_days shift so masked dates stay valid calendar
    dates (digit-FPE on a date string can produce e.g. month=19)."""
    h = hmac.new(master_key, f"date-offset|{context}".encode(), hashlib.sha256).digest()
    offset = (int.from_bytes(h[:4], "big") % (2 * max_days + 1)) - max_days
    return value + _dt.timedelta(days=offset)


def _is_null(value: Any) -> bool:
    """None, NaN, and pandas' pd.NA/pd.NaT sentinels -- checked by type
    name rather than importing pandas, so this module works fine in a
    polars-only environment too."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return type(value).__name__ in ("NAType", "NaTType")


def _mask_scalar(engine: MaskingEngine, value: Any, context: str) -> Any:
    if _is_null(value):
        return None
    if isinstance(value, bool):
        return value  # 2-value domain: no meaningful masking primitive
    if isinstance(value, (_dt.date, _dt.datetime)):
        return _perturb_date(engine.master_key, value, context)
    return engine.mask(str(value), context=context).text


def _mask_sample_rows(
    rows: list[dict], columns: list[ColumnInfo], policy: ColumnPolicy, engine: MaskingEngine, context: str
) -> list[dict]:
    sensitive_cols = {c.name for c in columns if policy.is_sensitive(c.name)}
    out = []
    for row in rows:
        out.append(
            {
                k: (_mask_scalar(engine, v, f"{context}:{k}") if k in sensitive_cols else v)
                for k, v in row.items()
            }
        )
    return out


def summarize_dataframe(
    df,
    *,
    context: str = "",
    policy: ColumnPolicy | None = None,
    mask_engine: MaskingEngine | None = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    numeric_bins: int = DEFAULT_NUMERIC_BINS,
    registry: SensitiveValueRegistry | None = None,
) -> dict:
    policy = policy or ColumnPolicy()
    engine = mask_engine or _default_engine()
    info = _adapt_frame(df, sample_rows=sample_rows)
    if registry is not None:
        registry.register_sample(info, policy)  # see registry.py: sample-only, not a full guarantee

    columns_out = []
    for c in info.columns:
        sensitive = policy.is_sensitive(c.name)
        entry: dict[str, Any] = {
            "name": c.name,
            "dtype": c.dtype,
            "kind": c.kind,
            "null_count": c.null_count,
            "sensitive": sensitive,
        }
        if sensitive:
            entry["distinct_count"] = (
                c.distinct_count
                if (c.distinct_count is not None and c.distinct_count <= LOW_CARDINALITY_LIMIT)
                else ("many" if c.distinct_count is not None else None)
            )
            if c.kind == "numeric":
                non_null = info.n_rows - c.null_count
                bins = numeric_quantile_bins(column_values(df, c.name), numeric_bins)
                entry["quantile_bins"] = bins
                if not bins and non_null > 0:
                    # too few non-null values for any bin to hide an individual
                    # record (see stats.numeric_quantile_bins) -- say so
                    # explicitly rather than let an empty list read as "no data"
                    entry["distribution_suppressed"] = (
                        f"only {non_null} non-null value(s); too few to report "
                        "a safe aggregate for a sensitive column"
                    )
        else:
            entry["distinct_count"] = c.distinct_count
            if c.kind == "numeric":
                entry.update(numeric_full_stats(column_values(df, c.name)))
            elif c.kind in ("string", "other") and c.distinct_count is not None and c.distinct_count <= LOW_CARDINALITY_LIMIT:
                vc = value_counts(column_values(df, c.name), LOW_CARDINALITY_LIMIT)
                if vc is not None:
                    entry["categories"] = vc
        columns_out.append(entry)

    return {
        "type": "dataframe",
        "shape": [info.n_rows, info.n_cols],
        "columns": columns_out,
        "sample_rows_masked": _mask_sample_rows(info.sample_rows, info.columns, policy, engine, context),
    }


def summarize_for_agent(
    obj: Any,
    *,
    context: str = "",
    policy: ColumnPolicy | None = None,
    mask_engine: MaskingEngine | None = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    numeric_bins: int = DEFAULT_NUMERIC_BINS,
    registry: SensitiveValueRegistry | None = None,
) -> dict:
    if is_dataframe(obj):
        return summarize_dataframe(
            obj,
            context=context,
            policy=policy,
            mask_engine=mask_engine,
            sample_rows=sample_rows,
            numeric_bins=numeric_bins,
            registry=registry,
        )

    engine = mask_engine or _default_engine()

    if _is_null(obj):
        return {"type": "NoneType", "value": None}
    if isinstance(obj, bool):
        return {"type": "bool", "value": obj}
    if isinstance(obj, (int, float)):
        # A bare numeric scalar out of a cell (df['salary'].mean(), a row
        # count, ...) is overwhelmingly an aggregate, not a single record's
        # raw value -- pass through. If a specific computation can leak an
        # individual value as a scalar (e.g. df['salary'].max()), that's a
        # column-level policy decision, not something summarize_for_agent
        # can see from a bare float; keep such reductions inside sensitive
        # code paths and prefer summarize_dataframe's binned stats instead.
        return {"type": type(obj).__name__, "value": obj}
    if isinstance(obj, str):
        return {"type": "str", "length": len(obj), "value_masked": engine.mask(obj, context=context).text}
    if isinstance(obj, (list, tuple)):
        preview = [
            _mask_scalar(engine, v, context) if isinstance(v, (str, _dt.date, _dt.datetime)) else v
            for v in list(obj)[:sample_rows]
        ]
        return {"type": type(obj).__name__, "length": len(obj), "preview_masked": preview}
    if isinstance(obj, dict):
        return {"type": "dict", "size": len(obj), "keys": list(obj.keys())[:100]}
    return {"type": type(obj).__name__, "repr_truncated": redact_text(repr(obj)[:200])}
