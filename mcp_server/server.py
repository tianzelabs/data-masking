"""MCP server exposing the fpe_mask engine as mask/unmask/classify tools.

Wire this in front of an agent so that raw sensitive text never has to be
handed to the model: a data-fetching tool masks its output through this
server before the result reaches the model's context, and any text the
model needs to write back out (e.g. into a follow-up tool call) is
unmasked here on the way out. See README.md for the two integration
patterns (explicit tool calls vs. PreToolUse/PostToolUse hooks).
"""
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from fpe_mask import MaskingEngine
from fpe_mask.classifier import classify_text
from fpe_mask.key_store import load_or_create_master_key
from fpe_mask.suffixes import HUNGARIAN_COMPANY_FORMS

_master_key = load_or_create_master_key()
_default_mode = os.environ.get("FPE_MASK_MODE", "standard")
_engines = {
    "standard": MaskingEngine(_master_key, mode="standard"),
    "strict": MaskingEngine(_master_key, mode="strict"),
}

server = MCPServer(
    name="fpe-mask",
    version="0.1.0",
    instructions=(
        "Format-preserving masking layer. Call mask_text on sensitive values "
        "BEFORE they are shown to the model; call unmask_text on masked "
        "values the model hands back before they go to a real downstream "
        "system. Always pass the same `context` string to both calls for a "
        "given field (e.g. the column/field name) -- it is not secret, but "
        "it must match or unmasking will silently produce garbage."
    ),
)


def _engine(mode: str | None) -> MaskingEngine:
    mode = mode or _default_mode
    try:
        return _engines[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}, expected 'standard' or 'strict'")


@server.tool()
def mask_text(text: str, context: str = "", mode: str | None = None) -> dict:
    """Format-preserving mask a string before it reaches the model.

    - `context`: public domain-separation tag (e.g. "company_name",
      "customer_email"). Not secret; must be passed back unchanged to
      unmask_text.
    - `mode`: "standard" (default) preserves case + letter/digit shape,
      combining plain and accented letters in one alphabet. "strict" also
      keeps accented vs. plain letters apart (e.g. Hungarian á/é/... stay in
      the same position as an accented letter), at the cost of falling
      back to a documented weaker substitution for classes that end up
      with very few or single occurrences -- see `singleton_fallback_classes`
      / `weak_domain_classes` in the response.
    """
    r = _engine(mode).mask(text, context=context)
    return {
        "masked": r.text,
        "weak_domain_classes": r.weak_domain_classes,
        "singleton_fallback_classes": r.singleton_fallback_classes,
        "is_fully_compliant": r.is_fully_compliant,
    }


@server.tool()
def unmask_text(text: str, context: str = "", mode: str | None = None) -> dict:
    """Reverse mask_text. `context` and `mode` must match the original call."""
    r = _engine(mode).unmask(text, context=context)
    return {"original": r.text}


@server.tool()
def classify_preview(text: str, mode: str | None = None) -> dict:
    """Debug helper: show how a string would be segmented, WITHOUT masking it.

    Returns the character-class tag for every character so you can check
    dictionary/segmentation behavior (e.g. that "Kft." is recognized as a
    literal suffix) without needing real sensitive data.
    """
    mode = mode or _default_mode
    atoms = classify_text(text, mode)
    return {
        "mode": mode,
        "atoms": [{"char": a.char, "class": a.class_name} for a in atoms],
    }


@server.tool()
def list_known_suffixes() -> dict:
    """List the legal-form tokens treated as literal (pass-through), e.g. Kft., Zrt."""
    return {"suffixes": sorted(HUNGARIAN_COMPANY_FORMS)}


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
