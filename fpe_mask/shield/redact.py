"""Redact stdout / exception text before it reaches the agent.

Two independent passes, both applied unconditionally:

1. Exact-match scrub against a SensitiveValueRegistry (catches "the actual
   cell value that appears verbatim in this traceback").
2. Generic high-confidence PII regexes (catches values from sources that
   were never registered -- a formatted string built inline, a value from
   an external join, etc.). This is NOT a full PII detector (no NER, no
   locale-specific ID formats beyond what's listed) -- it's a cheap second
   net, not a substitute for registering real data via the registry.

Never forward str(exception) / raw traceback text to the agent without
routing it through here first.
"""
from __future__ import annotations

import re
import traceback
from typing import Optional

from .registry import SensitiveValueRegistry

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("ssn_like", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone_like", re.compile(r"\+?\d[\d\-\s()]{8,}\d")),
    ("iban_like", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
]


def redact_text(text: str, registry: Optional[SensitiveValueRegistry] = None) -> str:
    out = text
    if registry is not None:
        for v in sorted(registry.all_values(), key=len, reverse=True):
            if v and v in out:
                out = out.replace(v, "<REDACTED>")
    for name, pattern in _PII_PATTERNS:
        out = pattern.sub(f"<REDACTED:{name}>", out)
    return out


def redact_exception(
    exc: BaseException, registry: Optional[SensitiveValueRegistry] = None, max_frames: int = 10
) -> dict:
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tail = tb_lines[-max_frames:] if len(tb_lines) > max_frames else tb_lines
    return {
        "type": type(exc).__name__,
        "message": redact_text(str(exc), registry),
        "traceback": [redact_text(line, registry) for line in tail],
    }
