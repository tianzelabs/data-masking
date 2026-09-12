"""Character-class alphabets used to segment and re-assemble masked text.

Every class below is a closed, ordered set of symbols. A run of input
characters that all belong to the same class is encrypted as one unit whose
output symbols are drawn from *that same class* -- this is what makes the
result "still look like a company name" / "still have a Hungarian letter in
the same slot" rather than a random blob.
"""
from __future__ import annotations

from dataclasses import dataclass

DIGIT = "0123456789"
LOWER_PLAIN = "abcdefghijklmnopqrstuvwxyz"
UPPER_PLAIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The 9 letters unique to the Hungarian alphabet beyond plain a-z (both the
# short and long accent variants of a/o/u count here as "accented").
LOWER_ACCENT_HU = "áéíóöőúüű"
UPPER_ACCENT_HU = "ÁÉÍÓÖŐÚÜŰ"

LOWER_COMBINED_HU = LOWER_PLAIN + LOWER_ACCENT_HU  # 35 symbols
UPPER_COMBINED_HU = UPPER_PLAIN + UPPER_ACCENT_HU  # 35 symbols


@dataclass(frozen=True)
class CharClass:
    name: str
    alphabet: str

    @property
    def radix(self) -> int:
        return len(self.alphabet)

    def __post_init__(self):
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError(f"alphabet for class {self.name!r} has duplicate symbols")

    def index(self, ch: str) -> int:
        return self.alphabet.index(ch)

    def symbol(self, i: int) -> str:
        return self.alphabet[i]

    def contains(self, ch: str) -> bool:
        return ch in self.alphabet


# --- "standard" mode: preserves case + "is a letter" / "is a digit", mixes
#     plain and accented letters of a given case into one alphabet. Every
#     real-world run is long enough to meet the NIST minimum domain size
#     (35**4 > 1_000_000), so this mode gets full, unqualified FF1 security
#     for anything but very short (<4 char) letter runs.
STANDARD_CLASSES: list[CharClass] = [
    CharClass("DIGIT", DIGIT),
    CharClass("LOWER", LOWER_COMBINED_HU),
    CharClass("UPPER", UPPER_COMBINED_HU),
]

# --- "strict" mode: also keeps plain vs. accented apart, so an accented
#     character always encrypts to another accented character in the same
#     case. The accent classes only have 9 symbols each -- see engine.py for
#     how the resulting small-domain problem is handled and documented.
STRICT_CLASSES: list[CharClass] = [
    CharClass("DIGIT", DIGIT),
    CharClass("LOWER_PLAIN", LOWER_PLAIN),
    CharClass("UPPER_PLAIN", UPPER_PLAIN),
    CharClass("LOWER_ACCENT_HU", LOWER_ACCENT_HU),
    CharClass("UPPER_ACCENT_HU", UPPER_ACCENT_HU),
]


def classify_char(ch: str, classes: list[CharClass]) -> CharClass | None:
    for c in classes:
        if c.contains(ch):
            return c
    return None
