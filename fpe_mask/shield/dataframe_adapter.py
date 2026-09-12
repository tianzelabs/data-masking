"""pandas / polars adapter: exposes one canonical interface so the rest of
the shield package doesn't care which dataframe library produced a result.

Only the extraction (column list, dtype, null/distinct counts, materializing
a column or a few sample rows) is backend-specific; all the actual
statistics math (quantile bins, value counts) runs on plain Python lists in
stats.py so it isn't duplicated per backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

EXACT_STATS_ROW_LIMIT = 5_000_000  # above this, skip nunique()-style exact counts


def is_pandas_df(obj: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(obj, pd.DataFrame)


def is_polars_df(obj: Any) -> bool:
    try:
        import polars as pl
    except ImportError:
        return False
    return isinstance(obj, pl.DataFrame)


def is_dataframe(obj: Any) -> bool:
    return is_pandas_df(obj) or is_polars_df(obj)


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    kind: str  # "numeric" | "string" | "bool" | "datetime" | "other"
    null_count: int
    distinct_count: int | None


@dataclass
class FrameInfo:
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]
    sample_rows: list[dict]


class _Backend(Protocol):
    def shape(self, df) -> tuple[int, int]: ...
    def columns(self, df) -> list[str]: ...
    def dtype_str(self, df, col: str) -> str: ...
    def kind(self, df, col: str) -> str: ...
    def null_count(self, df, col: str) -> int: ...
    def distinct_count(self, df, col: str) -> int | None: ...
    def column_values(self, df, col: str) -> list: ...
    def sample_rows(self, df, n: int) -> list[dict]: ...


class _PandasBackend:
    def shape(self, df):
        return df.shape

    def columns(self, df):
        return list(df.columns)

    def dtype_str(self, df, col):
        return str(df[col].dtype)

    def kind(self, df, col):
        import pandas as pd

        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            return "bool"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric"
        return "string"

    def null_count(self, df, col):
        return int(df[col].isna().sum())

    def distinct_count(self, df, col):
        if len(df) > EXACT_STATS_ROW_LIMIT:
            return None
        return int(df[col].nunique(dropna=True))

    def column_values(self, df, col):
        return df[col].dropna().tolist()

    def sample_rows(self, df, n):
        return df.head(n).to_dict(orient="records")


class _PolarsBackend:
    def shape(self, df):
        return (df.height, df.width)

    def columns(self, df):
        return list(df.columns)

    def dtype_str(self, df, col):
        return str(df.schema[col])

    def kind(self, df, col):
        import polars as pl

        dtype = df.schema[col]
        if dtype == pl.Boolean:
            return "bool"
        if dtype in (pl.Date, pl.Datetime) or str(dtype).startswith(("Date", "Datetime")):
            return "datetime"
        try:
            if dtype.is_numeric():
                return "numeric"
        except AttributeError:
            pass
        return "string"

    def null_count(self, df, col):
        return int(df[col].null_count())

    def distinct_count(self, df, col):
        if df.height > EXACT_STATS_ROW_LIMIT:
            return None
        return int(df[col].n_unique())

    def column_values(self, df, col):
        return df[col].drop_nulls().to_list()

    def sample_rows(self, df, n):
        return df.head(n).to_dicts()


def _backend_for(df) -> _Backend:
    if is_pandas_df(df):
        return _PandasBackend()
    if is_polars_df(df):
        return _PolarsBackend()
    raise TypeError(f"not a supported dataframe type: {type(df)!r}")


def adapt(df, sample_rows: int = 5) -> FrameInfo:
    backend = _backend_for(df)
    n_rows, n_cols = backend.shape(df)
    columns = [
        ColumnInfo(
            name=col,
            dtype=backend.dtype_str(df, col),
            kind=backend.kind(df, col),
            null_count=backend.null_count(df, col),
            distinct_count=backend.distinct_count(df, col),
        )
        for col in backend.columns(df)
    ]
    return FrameInfo(n_rows=n_rows, n_cols=n_cols, columns=columns, sample_rows=backend.sample_rows(df, sample_rows))


def column_values(df, col: str) -> list:
    """Materialize a full column as a plain Python list (non-null values
    only). Used for exact numeric stats and for registering full-column
    values with the redaction registry -- callers should be mindful this is
    O(n) memory for very large frames."""
    return _backend_for(df).column_values(df, col)
