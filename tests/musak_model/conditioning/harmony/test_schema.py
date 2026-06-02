from fractions import Fraction

import pytest
from pydantic import ValidationError

from musak_model.conditioning.harmony.schema import (
    HarmonicPlanWindow,
    harmonic_function_for_chord,
    harmonic_plan_windows_from_chord_windows,
)
from musak_model.harmony.decoding.schema import ChordWindow
from musak_model.harmony.schema import Chord, ChordQuality
from musak_shared.elements import HarmonicFunction


def test_harmonic_plan_window_derives_function_from_root_degree() -> None:
    window = HarmonicPlanWindow(
        start=Fraction(0),
        end=Fraction(1, 4),
        chord=Chord(root_degree=4, root_accidental=0, quality=ChordQuality.MAJOR),
    )

    assert window.harmonic_function == HarmonicFunction.PREDOMINANT


def test_harmonic_function_is_unknown_for_out_of_scale_root_degree() -> None:
    chord = Chord(root_degree=8, root_accidental=0, quality=ChordQuality.MAJOR)

    assert harmonic_function_for_chord(chord) is None


def test_harmonic_plan_window_rejects_empty_window() -> None:
    with pytest.raises(ValidationError, match="end must be greater than start"):
        HarmonicPlanWindow(
            start=Fraction(1, 4),
            end=Fraction(1, 4),
            chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
        )


def test_harmonic_plan_windows_convert_existing_chord_windows() -> None:
    chord_window = ChordWindow(
        start=Fraction(0),
        end=Fraction(1, 2),
        chord=Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR),
    )

    windows = harmonic_plan_windows_from_chord_windows((chord_window,))

    assert windows == (
        HarmonicPlanWindow(
            start=Fraction(0),
            end=Fraction(1, 2),
            chord=Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR),
        ),
    )
