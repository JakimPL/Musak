from fractions import Fraction

import pytest

from musak_shared.elements import is_dotted_duration, key_fifths_from_pitch_class, pitch_class_from_key_fifths


def test_is_dotted_duration_detects_binary_dotted_durations() -> None:
    assert is_dotted_duration(Fraction(3, 8))
    assert is_dotted_duration(Fraction(7, 16))


def test_is_dotted_duration_rejects_plain_and_tuplet_durations() -> None:
    assert not is_dotted_duration(Fraction(1, 4))
    assert not is_dotted_duration(Fraction(1, 3))


@pytest.mark.parametrize(
    ("key_fifths", "pitch_class"),
    [
        (0, 0),
        (1, 7),
        (2, 2),
        (-1, 5),
        (-2, 10),
        (7, 1),
        (-7, 11),
    ],
)
def test_pitch_class_from_key_fifths(key_fifths: int, pitch_class: int) -> None:
    assert pitch_class_from_key_fifths(key_fifths) == pitch_class


@pytest.mark.parametrize(
    ("pitch_class", "key_fifths"),
    [
        (0, 0),
        (7, 1),
        (2, 2),
        (5, -1),
        (10, -2),
        (1, -5),
        (11, 5),
    ],
)
def test_key_fifths_from_pitch_class(pitch_class: int, key_fifths: int) -> None:
    assert key_fifths_from_pitch_class(pitch_class) == key_fifths


def test_key_fifths_from_pitch_class_rejects_invalid_pitch_class() -> None:
    with pytest.raises(ValueError, match="pitch class"):
        key_fifths_from_pitch_class(12)
