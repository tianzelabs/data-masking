"""Local master-key provisioning for the MCP server / CLI.

This is a pragmatic default for running the plugin locally: a 32-byte key
is generated on first use and stored, mode-600, under ~/.data-masking (or
FPE_MASK_KEYFILE). For anything beyond local/dev use, replace this with a
real KMS (AWS KMS, GCP KMS, HashiCorp Vault, ...) -- swap out
`load_or_create_master_key()` for a call that fetches the key from there;
nothing else in this package needs to change.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

DEFAULT_KEY_PATH = Path(os.environ.get("FPE_MASK_KEYFILE", str(Path.home() / ".data-masking" / "master.key")))


def load_or_create_master_key(path: Path | None = None) -> bytes:
    path = path or DEFAULT_KEY_PATH
    if path.exists():
        data = path.read_bytes()
        if len(data) < 32:
            raise ValueError(f"key file {path} is corrupt (expected >=32 bytes, got {len(data)})")
        return data
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    path.write_bytes(key)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    return key
