"""Column sensitivity policy.

Default posture is conservative, per the workflow this was built for: every
column is treated as sensitive unless explicitly opted out. This means an
agent gets *no* raw values, no raw min/max, and no revealed categories for
any column until someone (a human, or a config file) deliberately marks
that specific column safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnPolicy:
    default_sensitive: bool = True
    safe_columns: set[str] = field(default_factory=set)
    forced_sensitive_columns: set[str] = field(default_factory=set)

    def is_sensitive(self, column: str) -> bool:
        if column in self.forced_sensitive_columns:
            return True
        if column in self.safe_columns:
            return False
        return self.default_sensitive

    def mark_safe(self, *columns: str) -> "ColumnPolicy":
        self.safe_columns.update(columns)
        self.forced_sensitive_columns.difference_update(columns)
        return self

    def mark_sensitive(self, *columns: str) -> "ColumnPolicy":
        self.forced_sensitive_columns.update(columns)
        self.safe_columns.difference_update(columns)
        return self
