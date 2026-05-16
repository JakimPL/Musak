from fractions import Fraction

from musak_shared.elements import is_dotted_duration


def test_is_dotted_duration_detects_binary_dotted_durations() -> None:
    assert is_dotted_duration(Fraction(3, 8))
    assert is_dotted_duration(Fraction(7, 16))


def test_is_dotted_duration_rejects_plain_and_tuplet_durations() -> None:
    assert not is_dotted_duration(Fraction(1, 4))
    assert not is_dotted_duration(Fraction(1, 3))
