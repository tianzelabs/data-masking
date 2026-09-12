import re

import pytest

from fpe_mask import MaskingEngine, generate_master_key
from fpe_mask.alphabets import LOWER_ACCENT_HU, UPPER_ACCENT_HU


KEY = b"0123456789abcdef0123456789abcdef"  # fixed 32-byte test key


def test_generate_master_key_length():
    assert len(generate_master_key()) == 32


@pytest.mark.parametrize("mode", ["standard", "strict"])
def test_roundtrip_hungarian_company_name(mode):
    engine = MaskingEngine(KEY, mode=mode)
    original = "Felmándi Kft."
    r = engine.mask(original, context="company_name")
    assert r.text != original
    assert len(r.text) == len(original)
    # legal-form suffix + punctuation + spacing preserved verbatim
    assert r.text.endswith("Kft.")
    assert r.text[8] == " "
    # roundtrip
    back = engine.unmask(r.text, context="company_name")
    assert back.text == original


def test_standard_mode_preserves_case_and_letter_digit_shape():
    engine = MaskingEngine(KEY, mode="standard")
    original = "Felmándi Kft."
    masked = engine.mask(original, context="c").text
    for o, m in zip(original, masked):
        if o.isupper():
            assert m.isupper()
        elif o.islower():
            assert m.islower()
        else:
            assert o == m  # space / punctuation untouched


def test_strict_mode_preserves_accent_class_position():
    engine = MaskingEngine(KEY, mode="strict")
    original = "Felmándi Kft."
    masked = engine.mask(original, context="c").text
    accent_positions = [i for i, ch in enumerate(original) if ch in LOWER_ACCENT_HU + UPPER_ACCENT_HU]
    assert accent_positions, "fixture should contain at least one accented letter"
    for i in accent_positions:
        assert masked[i] in LOWER_ACCENT_HU + UPPER_ACCENT_HU, (
            f"position {i} was accented ({original[i]!r}) but masked output "
            f"has {masked[i]!r}"
        )


def test_strict_mode_flags_singleton_fallback_for_lone_capital():
    # "F" is the only uppercase-plain char once "Kft." is recognized as a
    # literal suffix -- this must go through the documented weaker fallback,
    # and the engine must say so rather than silently claiming full FF1.
    engine = MaskingEngine(KEY, mode="strict")
    r = engine.mask("Felmándi Kft.", context="c")
    assert "UPPER_PLAIN" in r.singleton_fallback_classes
    assert not r.is_fully_compliant


def test_suffix_dictionary_case_insensitive_and_untouched():
    engine = MaskingEngine(KEY, mode="standard")
    for suffix in ["Kft.", "kft.", "ZRT.", "Bt."]:
        text = f"Almafa {suffix}"
        r = engine.mask(text, context="x")
        assert r.text.endswith(suffix)


def test_different_context_gives_different_ciphertext():
    engine = MaskingEngine(KEY, mode="standard")
    a = engine.mask("Felmándi Kft.", context="company_name").text
    b = engine.mask("Felmándi Kft.", context="billing_contact").text
    assert a != b


def test_roundtrip_many_random_hungarian_like_strings():
    import random

    random.seed(42)
    letters = "abcdefghijklmnopqrstuvwxyzáéíóöőúüűABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÖŐÚÜŰ0123456789 .-"
    engine = MaskingEngine(KEY, mode="strict")
    for _ in range(50):
        s = "".join(random.choice(letters) for _ in range(random.randint(3, 20)))
        r = engine.mask(s, context="field")
        back = engine.unmask(r.text, context="field")
        assert back.text == s
