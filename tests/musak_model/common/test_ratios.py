from fractions import Fraction

from musak_model.common.ratios import format_ratio


def test_format_ratio_accepts_fraction() -> None:
    assert format_ratio(Fraction(3, 8)) == "3/8"


def test_format_ratio_accepts_tuple() -> None:
    assert format_ratio((3, 4)) == "3/4"


def test_format_ratio_accepts_custom_separator() -> None:
    assert format_ratio(Fraction(3, 8), separator=":") == "3:8"
