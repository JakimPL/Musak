from dataclasses import dataclass
from fractions import Fraction

import pytest

from musak_model.harmony.windows import chord_window_grid


@dataclass(frozen=True)
class _WindowCase:
    measure_duration: Fraction
    total_duration: Fraction
    resolution: int
    expected: tuple[tuple[Fraction, Fraction], ...]


_CASES = (
    _WindowCase(Fraction(1), Fraction(2), 1, ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(2)))),
    _WindowCase(Fraction(1), Fraction(1), 2, ((Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(1)))),
    _WindowCase(
        Fraction(1),
        Fraction(1),
        4,
        (
            (Fraction(0), Fraction(1, 4)),
            (Fraction(1, 4), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(3, 4)),
            (Fraction(3, 4), Fraction(1)),
        ),
    ),
    _WindowCase(Fraction(3, 4), Fraction(3, 4), 2, ((Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(3, 4)))),
    _WindowCase(Fraction(5, 4), Fraction(5, 4), 1, ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(5, 4)))),
    _WindowCase(
        Fraction(1),
        Fraction(5, 2),
        1,
        ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(2)), (Fraction(2), Fraction(5, 2))),
    ),
)


@pytest.mark.parametrize("case", _CASES)
def test_chord_window_grid_tiles_bars_with_truncation(case: _WindowCase) -> None:
    windows = chord_window_grid(
        measure_duration=case.measure_duration,
        total_duration=case.total_duration,
        resolution=case.resolution,
    )

    assert windows == case.expected
