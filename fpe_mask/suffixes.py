"""Dictionary of legal-form tokens that should pass through masking untouched.

These are not personally identifying -- they're a small closed enum (a
Hungarian "Kft." is a legal form shared by ~450,000 companies) -- so
encrypting them destroys downstream utility (the agent needs to know "this
is a limited-liability company") without buying any privacy. They are
matched whole-word, case-insensitively, with an optional trailing period,
and passed through with their original casing preserved.

Extend HUNGARIAN_COMPANY_FORMS (or add your own list and pass it to
Classifier(extra_suffixes=...)) for other jurisdictions.
"""
from __future__ import annotations

HUNGARIAN_COMPANY_FORMS: frozenset[str] = frozenset(
    s.lower()
    for s in [
        "Kft",  # Korlátolt Felelősségű Társaság - LLC
        "Zrt",  # Zártkörűen Működő Részvénytársaság - private co. ltd by shares
        "Nyrt",  # Nyilvánosan Működő Részvénytársaság - public co. ltd by shares
        "Bt",  # Betéti Társaság - limited partnership
        "Kkt",  # Közkereseti Társaság - general partnership
        "Kv",  # Közös Vállalat - joint enterprise
        "Ev",  # Egyéni Vállalkozó - sole trader
        "Kht",  # Közhasznú Társaság - legacy nonprofit company
        "Rt",  # Részvénytársaság - legacy (pre Zrt/Nyrt split)
        "Szövetkezet",  # Cooperative
        "Egyesület",  # Association
        "Alapítvány",  # Foundation
    ]
)


def is_known_suffix(word: str, extra: frozenset[str] = frozenset()) -> bool:
    stripped = word.rstrip(".").lower()
    return stripped in HUNGARIAN_COMPANY_FORMS or stripped in extra
