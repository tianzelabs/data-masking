"""FF1 format-preserving encryption (NIST SP 800-38G, Algorithm 7/8).

Clean-room implementation from the published specification, cross-checked
round-by-round against the official NIST FF1 sample vectors
(FF1samples.pdf, Sample #1: AES-128 key 2B7E151628AED2A6ABF7158809CF4F3C,
radix=10, empty tweak, PT=0123456789 -> CT=2433477484). See tests/test_ff1.py.

This module implements the *primitive* faithfully, including its documented
minimum-domain requirement (radix**length >= 1_000_000) and minimum length
requirement (length >= 2). Callers that need to encrypt values which don't
meet those bounds (e.g. a 1-character alphabet run) must not call this
class directly with allow_weak_domain=True lightly -- see fpe_mask.engine
for how the rest of this package handles that case honestly instead of
pretending the primitive gives real security where the domain is tiny.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

MIN_RADIX = 2
MAX_RADIX = 1 << 16
MIN_DOMAIN_SIZE = 1_000_000  # radix**length lower bound mandated by SP 800-38G


class FF1Error(ValueError):
    pass


class DomainTooSmallError(FF1Error):
    """Raised when radix**length < MIN_DOMAIN_SIZE and allow_weak_domain=False.

    This is not a bug to "fix" by relaxing the check silently: for a small
    alphabet (e.g. 9 Hungarian accented vowels), no cipher construction can
    give the input real security -- the entropy ceiling is set by the
    alphabet size itself (log2(9!) ~= 18.5 bits total). Passing
    allow_weak_domain=True is an explicit, auditable acknowledgement of that,
    not a fix for it.
    """


def _bytestring(n: int, x: int) -> bytes:
    return x.to_bytes(n, "big")


def _num_radix(digits: list[int], radix: int) -> int:
    n = 0
    for d in digits:
        n = n * radix + d
    return n


def _str_radix(x: int, radix: int, length: int) -> list[int]:
    out = [0] * length
    for i in range(length - 1, -1, -1):
        out[i] = x % radix
        x //= radix
    return out


class _AesEcb:
    """Single-block AES-ECB encryption, used as the building block for the
    CBC-MAC PRF and the counter-mode expansion that FF1 specifies."""

    def __init__(self, key: bytes):
        if len(key) not in (16, 24, 32):
            raise FF1Error("key must be 16, 24, or 32 bytes (AES-128/192/256)")
        self._cipher = Cipher(algorithms.AES(key), modes.ECB())

    def encrypt_block(self, block: bytes) -> bytes:
        assert len(block) == 16
        enc = self._cipher.encryptor()
        out = enc.update(block) + enc.finalize()
        return out


@dataclass
class FF1:
    key: bytes
    radix: int
    allow_weak_domain: bool = False

    def __post_init__(self):
        if not (MIN_RADIX <= self.radix <= MAX_RADIX):
            raise FF1Error(f"radix {self.radix} out of range [{MIN_RADIX}, {MAX_RADIX}]")
        self._aes = _AesEcb(self.key)

    # -- domain checks --------------------------------------------------
    def check_length(self, n: int) -> None:
        if n < 2:
            raise FF1Error(
                f"FF1 requires length >= 2 (got {n}); a single symbol cannot be "
                "Feistel-split. Pool it with other same-class symbols or use a "
                "different fallback for isolated singletons."
            )
        if not self.allow_weak_domain and self.radix**n < MIN_DOMAIN_SIZE:
            raise DomainTooSmallError(
                f"radix**length = {self.radix}**{n} = {self.radix**n} < "
                f"{MIN_DOMAIN_SIZE} (NIST SP 800-38G minimum domain size). "
                "Pass allow_weak_domain=True to proceed anyway (only meaningful "
                "for defense-in-depth / cosmetic use, not as sole protection)."
            )

    # -- core round function ---------------------------------------------
    def _prf(self, data: bytes) -> bytes:
        assert len(data) % 16 == 0
        chain = bytes(16)
        for i in range(0, len(data), 16):
            block = data[i : i + 16]
            x = bytes(a ^ b for a, b in zip(chain, block))
            chain = self._aes.encrypt_block(x)
        return chain

    def _prp_expand(self, r: bytes, d: int) -> bytes:
        out = bytearray(r)
        j = 1
        r_int = int.from_bytes(r, "big")
        while len(out) < d:
            block = (r_int ^ j).to_bytes(16, "big")
            out += self._aes.encrypt_block(block)
            j += 1
        return bytes(out[:d])

    def _params(self, n: int, tweak: bytes):
        u = n // 2
        v = n - u
        b = math.ceil(math.ceil(v * math.log2(self.radix)) / 8)
        d = 4 * math.ceil(b / 4) + 4
        t = len(tweak)
        p = (
            bytes([1, 2, 1])
            + _bytestring(3, self.radix)
            + bytes([10])
            + _bytestring(1, u % 256)
            + _bytestring(4, n)
            + _bytestring(4, t)
        )
        return u, v, b, d, p

    def encrypt(self, digits: list[int], tweak: bytes = b"") -> list[int]:
        n = len(digits)
        self.check_length(n)
        u, v, b, d, p = self._params(n, tweak)
        t = len(tweak)
        a = digits[:u]
        bb = digits[u:]
        pad_len = (-t - b - 1) % 16
        for i in range(10):
            q = tweak + bytes(pad_len) + bytes([i]) + _bytestring(b, _num_radix(bb, self.radix))
            r = self._prf(p + q)
            s = self._prp_expand(r, d)
            y = int.from_bytes(s, "big")
            m = u if i % 2 == 0 else v
            c = (_num_radix(a, self.radix) + y) % (self.radix**m)
            c_digits = _str_radix(c, self.radix, m)
            a, bb = bb, c_digits
        return a + bb

    def decrypt(self, digits: list[int], tweak: bytes = b"") -> list[int]:
        n = len(digits)
        self.check_length(n)
        u, v, b, d, p = self._params(n, tweak)
        t = len(tweak)
        a = digits[:u]
        bb = digits[u:]
        pad_len = (-t - b - 1) % 16
        for i in range(9, -1, -1):
            m = u if i % 2 == 0 else v
            q = tweak + bytes(pad_len) + bytes([i]) + _bytestring(b, _num_radix(a, self.radix))
            r = self._prf(p + q)
            s = self._prp_expand(r, d)
            y = int.from_bytes(s, "big")
            c = (_num_radix(bb, self.radix) - y) % (self.radix**m)
            c_digits = _str_radix(c, self.radix, m)
            bb, a = a, c_digits
        return a + bb
