#!/usr/bin/env python3
"""OPTIONAL, best-effort PostToolUse hook: auto-mask specific fields in a
tool's JSON output before it reaches the model, instead of relying on the
agent to remember to call mask_text itself.

This is a reference sketch, not a verified integration: Claude Code's hook
stdin/stdout JSON contract has changed across versions, so treat the
`hook_input.get(...)` calls below as the part to adjust for your version
(check `claude --help hooks` / current docs). The masking logic itself
(fpe_mask) is what's actually tested -- see tests/.

Configure in settings.json, e.g.:
  "hooks": {
    "PostToolUse": [{
      "matcher": "query_customer_db",
      "hooks": [{"type": "command", "command": "python3 hooks/mask_tool_io.py"}]
    }]
  }

FPE_MASK_FIELDS selects which JSON keys in the tool response get masked,
e.g. FPE_MASK_FIELDS="company_name,contact_name".
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpe_mask import MaskingEngine
from fpe_mask.key_store import load_or_create_master_key

FIELDS = {f.strip() for f in os.environ.get("FPE_MASK_FIELDS", "").split(",") if f.strip()}
MODE = os.environ.get("FPE_MASK_MODE", "standard")


def _mask_value(engine: MaskingEngine, key: str, value):
    if isinstance(value, str):
        return engine.mask(value, context=key).text
    if isinstance(value, dict):
        return {k: (_mask_value(engine, k, v) if k in FIELDS else v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_value(engine, key, v) for v in value]
    return value


def main() -> int:
    hook_input = json.load(sys.stdin)
    engine = MaskingEngine(load_or_create_master_key(), mode=MODE)

    # Adjust this key path to match your Claude Code version's PostToolUse
    # payload shape -- this assumes `tool_response` holds the raw tool
    # output (string or dict).
    response = hook_input.get("tool_response")
    if response is None:
        print(json.dumps({}))
        return 0

    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            print(json.dumps({}))
            return 0
        masked = _mask_value(engine, "", parsed)
        print(json.dumps({"hookSpecificOutput": {"updatedToolResponse": json.dumps(masked)}}))
    elif isinstance(response, dict):
        masked = _mask_value(engine, "", response)
        print(json.dumps({"hookSpecificOutput": {"updatedToolResponse": masked}}))
    else:
        print(json.dumps({}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
