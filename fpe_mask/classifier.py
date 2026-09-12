"""Tag every character of the input with the class it belongs to.

The key invariant the rest of the package depends on: running this same
function over the *masked* output reproduces exactly the same tagging as
over the original plaintext (masking never changes a character's class or
which characters are literal), so unmasking never needs any side-channel
segmentation metadata -- it just re-classifies the ciphertext.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .alphabets import CharClass, STANDARD_CLASSES, STRICT_CLASSES, classify_char
from .suffixes import is_known_suffix

MODES: dict[str, list[CharClass]] = {
    "standard": STANDARD_CLASSES,
    "strict": STRICT_CLASSES,
}

_WORD_RE = re.compile(r"\S+")


@dataclass
class Atom:
    char: str
    class_name: str | None  # None means literal / pass-through


def classes_for_mode(mode: str) -> list[CharClass]:
    try:
        return MODES[mode]
    except KeyError as e:
        raise ValueError(f"unknown mode {mode!r}, expected one of {list(MODES)}") from e


def class_by_name(mode: str, name: str) -> CharClass:
    for c in classes_for_mode(mode):
        if c.name == name:
            return c
    raise KeyError(name)


def classify_text(text: str, mode: str, extra_suffixes: frozenset[str] = frozenset()) -> list[Atom]:
    classes = classes_for_mode(mode)
    suffix_mask = bytearray(len(text))
    for m in _WORD_RE.finditer(text):
        if is_known_suffix(m.group(0), extra_suffixes):
            for i in range(m.start(), m.end()):
                suffix_mask[i] = 1

    atoms: list[Atom] = []
    for i, ch in enumerate(text):
        if suffix_mask[i]:
            atoms.append(Atom(ch, None))
            continue
        c = classify_char(ch, classes)
        atoms.append(Atom(ch, c.name if c else None))
    return atoms
