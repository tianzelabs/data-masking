import pytest

from fpe_mask.ff1 import FF1, DomainTooSmallError, FF1Error


def _key():
    # AES-128 key from the official NIST FF1 sample vectors document
    return bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")


def test_nist_sample_1_encrypt():
    ff1 = FF1(key=_key(), radix=10)
    pt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ct = ff1.encrypt(pt, tweak=b"")
    assert "".join(str(d) for d in ct) == "2433477484"


def test_nist_sample_1_decrypt_roundtrip():
    ff1 = FF1(key=_key(), radix=10)
    ct = [2, 4, 3, 3, 4, 7, 7, 4, 8, 4]
    pt = ff1.decrypt(ct, tweak=b"")
    assert "".join(str(d) for d in pt) == "0123456789"


@pytest.mark.parametrize("radix,length", [(10, 10), (26, 8), (35, 6), (62, 5)])
def test_roundtrip_random(radix, length):
    import os

    ff1 = FF1(key=os.urandom(16), radix=radix, allow_weak_domain=True)
    for _ in range(20):
        digits = [int.from_bytes(os.urandom(1), "big") % radix for _ in range(length)]
        ct = ff1.encrypt(digits, tweak=b"ctx")
        pt = ff1.decrypt(ct, tweak=b"ctx")
        assert pt == digits


def test_different_tweak_changes_ciphertext():
    ff1 = FF1(key=_key(), radix=10)
    pt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ct1 = ff1.encrypt(pt, tweak=b"a")
    ct2 = ff1.encrypt(pt, tweak=b"b")
    assert ct1 != ct2


def test_length_below_minimum_rejected():
    ff1 = FF1(key=_key(), radix=10, allow_weak_domain=True)
    with pytest.raises(FF1Error):
        ff1.encrypt([5])


def test_small_domain_rejected_by_default():
    ff1 = FF1(key=_key(), radix=9)  # 9 Hungarian accented vowels
    with pytest.raises(DomainTooSmallError):
        ff1.encrypt([0, 1, 2, 3, 4, 5])  # 9**6 = 531441 < 1_000_000


def test_small_domain_allowed_when_opted_in():
    ff1 = FF1(key=_key(), radix=9, allow_weak_domain=True)
    ct = ff1.encrypt([0, 1], tweak=b"x")
    pt = ff1.decrypt(ct, tweak=b"x")
    assert pt == [0, 1]
