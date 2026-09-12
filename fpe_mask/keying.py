"""Key derivation and the small-domain fallback permutation.

Two distinct primitives live here:

- derive_subkey: standard HKDF-SHA256 domain separation, one independent
  AES key per character class, so no key material is ever reused across
  classes.
- keyed_permutation: a deterministic, keyed bijection on {0..size-1} built
  from an HMAC keystream (Fisher-Yates). This is used ONLY for singleton
  runs that are structurally too short for FF1 (length 1) -- it is a plain
  substitution cipher, not a NIST-grade FPE construction, and its real
  security ceiling is log2(size!) bits (e.g. ~18.5 bits for the 9 Hungarian
  accent letters). engine.py surfaces every use of this path in
  MaskResult.singleton_fallback_classes so callers can see where the
  weaker guarantee applies.
"""
from __future__ import annotations

import hashlib
import hmac

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def derive_subkey(master_key: bytes, info: bytes, length: int = 16) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(master_key)


def _hmac_stream(key: bytes, tag: bytes, nbytes: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < nbytes:
        out += hmac.new(key, tag + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        counter += 1
    return out[:nbytes]


def keyed_permutation(key: bytes, tag: bytes, size: int) -> tuple[list[int], list[int]]:
    """Return (perm, inverse) such that perm is a deterministic keyed
    shuffle of range(size). perm[i] is what plaintext symbol i maps to."""
    perm = list(range(size))
    # Fisher-Yates using an HMAC keystream for the random indices.
    stream = _hmac_stream(key, tag, size * 4)
    for i in range(size - 1, 0, -1):
        r = int.from_bytes(stream[i * 4 : i * 4 + 4], "big")
        j = r % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    inverse = [0] * size
    for i, p in enumerate(perm):
        inverse[p] = i
    return perm, inverse
