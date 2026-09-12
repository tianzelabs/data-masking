"""Agent-facing execution shield: decide what a code-executing agent gets
back after running a cell, given that raw data must stay confined to the
human-facing display (e.g. a marimo canvas) and never enter the model's
context.

    from fpe_mask.shield import summarize_for_agent, ColumnPolicy, SensitiveValueRegistry, redact_exception

    policy = ColumnPolicy().mark_safe("status", "country")  # opt specific columns OUT of the conservative default
    registry = SensitiveValueRegistry()
    registry.register_dataframe(df, policy)   # do this once when data is loaded

    # after running a cell and getting back `result` (or catching `exc`):
    agent_result = summarize_for_agent(result, context="customers_table", policy=policy, registry=registry)
    # on error instead:
    agent_error = redact_exception(exc, registry=registry)

    # meanwhile the *unfiltered* `result` still goes to marimo's normal
    # cell-output rendering for the human viewing the canvas.
"""
from .policy import ColumnPolicy
from .redact import redact_exception, redact_text
from .registry import SensitiveValueRegistry
from .summarize import summarize_dataframe, summarize_for_agent

__all__ = [
    "ColumnPolicy",
    "SensitiveValueRegistry",
    "redact_exception",
    "redact_text",
    "summarize_dataframe",
    "summarize_for_agent",
]
