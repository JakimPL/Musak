from fractions import Fraction

import pytest

from musak_shared.ratios import format_ratio, parse_ratio


def test_format_ratio_accepts_fraction() -> None:
    assert format_ratio(Fraction(3, 8)) == "3/8"


def test_format_ratio_accepts_tuple() -> None:
    assert format_ratio((3, 4)) == "3/4"


def test_format_ratio_accepts_custom_separator() -> None:
    assert format_ratio(Fraction(3, 8), separator=":") == "3:8"


@pytest.mark.parametrize("text", ["3/8", "3:8"])
def test_parse_ratio_accepts_supported_separators(text: str) -> None:
    assert parse_ratio(text) == Fraction(3, 8)


def test_parse_ratio_rejects_invalid_text() -> None:
    with pytest.raises(ValueError, match="separator"):
        parse_ratio("3")
