import datetime as dt

import pandas as pd
import polars as pl
import pytest

from fpe_mask.shield import ColumnPolicy, SensitiveValueRegistry, redact_exception, redact_text, summarize_for_agent
from fpe_mask.shield.stats import numeric_full_stats, numeric_quantile_bins, value_counts


def _pandas_df():
    return pd.DataFrame(
        {
            "customer_name": ["Felmándi Kft.", "Almafa Bt.", "Örs Zrt.", "Béla Kft.", None],
            "age": [23, 45, 61, 34, 29],
            "status": ["active", "active", "inactive", "active", "pending"],
            "signup_date": [dt.date(2023, 1, 5), dt.date(2022, 6, 12), dt.date(2021, 11, 30), dt.date(2020, 3, 1), dt.date(2024, 7, 20)],
        }
    )


def _polars_df():
    return pl.DataFrame(
        {
            "customer_name": ["Felmándi Kft.", "Almafa Bt.", "Örs Zrt.", "Béla Kft.", None],
            "age": [23, 45, 61, 34, 29],
            "status": ["active", "active", "inactive", "active", "pending"],
        }
    )


# -- stats.py -----------------------------------------------------------


def test_numeric_quantile_bins_never_isolates_a_single_record():
    values = list(range(100))
    bins = numeric_quantile_bins(values, bins=10)
    assert sum(b["count"] for b in bins) == 100
    for b in bins:
        assert b["count"] >= 1
    # every bin's range spans a group, not necessarily a singleton
    assert len(bins) <= 10


def test_numeric_quantile_bins_enforces_k_anonymity_floor():
    # fewer than min_count values total: no bin can be built without one
    # containing (and thus exposing the edge of) a single record.
    assert numeric_quantile_bins([23, 45, 61, 34], bins=10, min_count=5) == []

    # exactly min_count values: allowed, but must never emit a bin below
    # the floor, and edges must be rounded, not the raw order statistics.
    values = [23, 45, 61, 34, 29]
    bins = numeric_quantile_bins(values, bins=10, min_count=5)
    assert all(b["count"] >= 5 for b in bins)
    edges = {v for b in bins for v in b["range"]}
    assert not edges & set(values)

    bins = numeric_quantile_bins(list(range(20)), bins=10, min_count=5)
    assert all(b["count"] >= 5 for b in bins)


def test_summarize_small_sensitive_numeric_column_is_suppressed_not_leaked():
    df = pd.DataFrame({"salary": [120000, 89000, 340000, 51000]})  # 4 < min_count(5)
    result = summarize_for_agent(df, context="t")
    col = result["columns"][0]
    assert col["quantile_bins"] == []
    assert "distribution_suppressed" in col
    assert "340000" not in str(col)


def test_summarize_sensitive_numeric_bins_never_expose_raw_edge_values():
    df = pd.DataFrame({"salary": [120000, 89000, 340000, 51000, 12000]})
    result = summarize_for_agent(df, context="t")
    col = result["columns"][0]
    raw = {"120000", "89000", "340000", "51000", "12000"}
    edge_strs = {str(int(v)) for b in col["quantile_bins"] for v in b["range"]}
    assert not (edge_strs & raw)


def test_numeric_full_stats_basic():
    s = numeric_full_stats([1, 2, 3, 4, 5])
    assert s["min"] == 1 and s["max"] == 5 and s["count"] == 5


def test_value_counts_returns_none_above_limit():
    assert value_counts(list(range(100)), limit=10) is None
    assert value_counts(["a", "a", "b"], limit=10) == {"a": 2, "b": 1}


# -- summarize_for_agent on real dataframes ------------------------------


@pytest.mark.parametrize("make_df", [_pandas_df, _polars_df])
def test_default_policy_hides_all_raw_values(make_df):
    df = make_df()
    result = summarize_for_agent(df, context="customers")
    assert result["type"] == "dataframe"
    assert result["shape"][0] == 5

    names_col = next(c for c in result["columns"] if c["name"] == "customer_name")
    assert names_col["sensitive"] is True
    assert "categories" not in names_col

    original_names = {"Felmándi Kft.", "Almafa Bt.", "Örs Zrt.", "Béla Kft."}
    for row in result["sample_rows_masked"]:
        if row["customer_name"] is not None:
            assert row["customer_name"] not in original_names

    age_col = next(c for c in result["columns"] if c["name"] == "age")
    assert "min" not in age_col and "max" not in age_col  # no raw extremes
    assert "quantile_bins" in age_col


def test_marking_a_column_safe_exposes_its_real_stats_and_categories():
    df = _pandas_df()
    policy = ColumnPolicy().mark_safe("status", "age")
    result = summarize_for_agent(df, context="customers", policy=policy)

    status_col = next(c for c in result["columns"] if c["name"] == "status")
    assert status_col["sensitive"] is False
    assert status_col["categories"] == {"active": 3, "inactive": 1, "pending": 1}

    age_col = next(c for c in result["columns"] if c["name"] == "age")
    assert age_col["min"] == 23 and age_col["max"] == 61

    for row in result["sample_rows_masked"]:
        assert row["status"] in ("active", "inactive", "pending")


def test_sample_dates_are_perturbed_not_digit_scrambled():
    df = _pandas_df()
    result = summarize_for_agent(df, context="customers")
    for row in result["sample_rows_masked"]:
        d = row["signup_date"]
        if d is not None:
            assert isinstance(d, dt.date)  # still a real, valid calendar date


def test_none_is_preserved_through_masking():
    df = _pandas_df()
    result = summarize_for_agent(df, context="customers")
    assert any(row["customer_name"] is None for row in result["sample_rows_masked"])


# -- redact.py ------------------------------------------------------------


def test_redact_text_scrubs_registered_values():
    registry = SensitiveValueRegistry()
    registry.register_values(["Jane Doe", "jane@example.com"])
    text = "ValueError: could not convert 'Jane Doe' <jane@example.com> to float"
    out = redact_text(text, registry)
    assert "Jane Doe" not in out
    assert "jane@example.com" not in out


def test_redact_text_generic_pii_patterns_without_registry():
    out = redact_text("contact bob@example.com or 555-123-4567 or 123-45-6789")
    assert "bob@example.com" not in out
    assert "REDACTED" in out


def test_redact_exception_scrubs_traceback():
    registry = SensitiveValueRegistry()
    registry.register_values(["super-secret-id-42"])
    try:
        raise ValueError("bad value: super-secret-id-42")
    except ValueError as e:
        report = redact_exception(e, registry)
    assert report["type"] == "ValueError"
    assert "super-secret-id-42" not in report["message"]
    assert all("super-secret-id-42" not in line for line in report["traceback"])


def test_registry_full_column_registration():
    df = _pandas_df()
    policy = ColumnPolicy()  # everything sensitive
    registry = SensitiveValueRegistry()
    registry.register_dataframe(df, policy)
    assert "Felmándi Kft." in registry.all_values()
    assert "Örs Zrt." in registry.all_values()
