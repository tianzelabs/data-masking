"""Known-value registry for exact-match redaction of stdout/tracebacks.

Schema-level summaries (stats.py) are the main defense for *intentional*
output. This registry is the second line of defense for *accidental*
leakage -- e.g. a traceback that happens to embed the value that caused a
parse error (`ValueError: could not convert 'Jane Doe' to float`).

IMPORTANT: only registers what it's told about. Call
`register_dataframe(df, policy)` (materializes and registers FULL sensitive
columns) whenever a dataframe is loaded into the notebook, not just at
result-summarization time -- summarize_for_agent() only auto-registers the
small `head()` sample it shows, which protects a traceback that mentions a
sampled row but NOT one that mentions row #48213. Full registration is an
explicit, deliberate call because it's O(n) memory/time on the sensitive
columns; do it once per load, not per cell execution.
"""
from __future__ import annotations

from typing import Any, Iterable

from .dataframe_adapter import FrameInfo, column_values, is_dataframe
from .policy import ColumnPolicy


class SensitiveValueRegistry:
    def __init__(self, min_len: int = 2, max_len: int = 200, max_values: int = 500_000):
        self._values: set[str] = set()
        self.min_len = min_len
        self.max_len = max_len
        self.max_values = max_values

    def __len__(self) -> int:
        return len(self._values)

    def register_values(self, values: Iterable[Any]) -> None:
        for v in values:
            if v is None or len(self._values) >= self.max_values:
                continue
            s = str(v)
            if self.min_len <= len(s) <= self.max_len:
                self._values.add(s)

    def register_sample(self, info: FrameInfo, policy: ColumnPolicy) -> None:
        sensitive = {c.name for c in info.columns if policy.is_sensitive(c.name)}
        for row in info.sample_rows:
            self.register_values(v for k, v in row.items() if k in sensitive)

    def register_dataframe(self, df, policy: ColumnPolicy) -> None:
        """Full-column registration -- see module docstring. Prefer this
        over relying on register_sample for anything beyond a demo."""
        if not is_dataframe(df):
            raise TypeError(f"not a supported dataframe type: {type(df)!r}")
        for name in _columns_of(df):
            if policy.is_sensitive(name):
                self.register_values(column_values(df, name))

    def all_values(self) -> list[str]:
        return list(self._values)


def _columns_of(df) -> list[str]:
    from .dataframe_adapter import is_pandas_df, is_polars_df

    if is_pandas_df(df):
        return list(df.columns)
    if is_polars_df(df):
        return list(df.columns)
    raise TypeError(f"not a supported dataframe type: {type(df)!r}")
