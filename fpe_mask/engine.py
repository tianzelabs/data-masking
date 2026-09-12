"""Top-level masking engine: ties classifier + FF1 + singleton fallback together.

    engine = MaskingEngine(master_key, mode="strict")
    r = engine.mask("Felmándi Kft.", context="company_name")
    r.text                      # e.g. "Őyzqtbo Kft."   (still Title Case,
                                 #  still ends in "Kft.", position 2 is
                                 #  still one of the accented letters)
    engine.unmask(r.text, context="company_name").text == "Felmándi Kft."

`context` is public (not secret) domain-separation input -- pass the same
value to mask() and unmask() for a given field/column so that the same
name encrypts differently in different fields, and always pass it back
unchanged or unmasking will silently produce garbage (not an error: FF1
with a wrong tweak still "succeeds", it just decrypts to nonsense).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .alphabets import CharClass
from .classifier import Atom, class_by_name, classes_for_mode, classify_text
from .ff1 import FF1, DomainTooSmallError
from .keying import derive_subkey, keyed_permutation


def generate_master_key() -> bytes:
    """32 bytes of CSPRNG output, suitable as MaskingEngine(master_key=...)."""
    return os.urandom(32)


@dataclass
class MaskResult:
    text: str
    weak_domain_classes: list[str] = field(default_factory=list)
    singleton_fallback_classes: list[str] = field(default_factory=list)

    @property
    def is_fully_compliant(self) -> bool:
        """False if any class in this call fell back to a weaker-than-NIST
        guarantee (tiny domain accepted, or a length-1 substitution)."""
        return not self.weak_domain_classes and not self.singleton_fallback_classes


class MaskingEngine:
    def __init__(
        self,
        master_key: bytes,
        mode: str = "standard",
        extra_suffixes: frozenset[str] = frozenset(),
    ):
        if len(master_key) < 32:
            raise ValueError("master_key should be >= 32 bytes of real entropy (see generate_master_key())")
        self.master_key = master_key
        self.mode = mode
        self.extra_suffixes = extra_suffixes
        self._classes = classes_for_mode(mode)
        self._subkeys = {c.name: derive_subkey(master_key, f"ff1:{c.name}".encode()) for c in self._classes}
        self._perm_key = derive_subkey(master_key, b"singleton-perm:v1", length=32)

    def mask(self, text: str, context: str = "") -> MaskResult:
        return self._transform(text, context, forward=True)

    def unmask(self, text: str, context: str = "") -> MaskResult:
        return self._transform(text, context, forward=False)

    # -- internals --------------------------------------------------------
    def _transform(self, text: str, context: str, forward: bool) -> MaskResult:
        atoms: list[Atom] = classify_text(text, self.mode, self.extra_suffixes)
        groups: dict[str, list[int]] = {}
        for idx, a in enumerate(atoms):
            if a.class_name is not None:
                groups.setdefault(a.class_name, []).append(idx)

        out_chars = [a.char for a in atoms]
        weak_domain: list[str] = []
        singleton_fallback: list[str] = []

        for class_name, idxs in groups.items():
            cls = class_by_name(self.mode, class_name)
            digits = [cls.index(atoms[i].char) for i in idxs]
            tweak = f"{context}|{class_name}".encode("utf-8")

            if len(digits) == 1:
                singleton_fallback.append(class_name)
                perm, inv = keyed_permutation(self._perm_key, b"singleton|" + tweak, cls.radix)
                table = perm if forward else inv
                out_chars[idxs[0]] = cls.symbol(table[digits[0]])
                continue

            key = self._subkeys[class_name]
            ff1 = FF1(key=key, radix=cls.radix)
            op = ff1.encrypt if forward else ff1.decrypt
            try:
                result = op(digits, tweak)
            except DomainTooSmallError:
                weak_domain.append(class_name)
                ff1_weak = FF1(key=key, radix=cls.radix, allow_weak_domain=True)
                op_weak = ff1_weak.encrypt if forward else ff1_weak.decrypt
                result = op_weak(digits, tweak)

            for i, d in zip(idxs, result):
                out_chars[i] = cls.symbol(d)

        return MaskResult("".join(out_chars), weak_domain, singleton_fallback)
