from fractions import Fraction

from musak_model.data.scale_match import match_scale
from musak_model.data.schema import ParsedBar, ParsedNote
from musak_model.tokens.schema import ScaleType


def test_matches_major_scale_from_duration_weighted_distribution_without_key_signature() -> None:
    match = match_scale(
        [_bar([62, 64, 66, 67, 69, 71, 73])],
        [_bar([])],
        minimum_in_scale_weight_fraction=0.9,
        minimum_best_margin=0.03,
    )

    assert match.scale_root == 2
    assert match.scale_type == ScaleType.MAJOR
    assert match.diagnostics.in_scale_weight_fraction == 1.0


def test_wrong_declared_key_is_overridden_by_pitch_distribution() -> None:
    match = match_scale(
        [_bar([62, 64, 66, 67, 69, 71, 73], declared_key_fifths=0)],
        [_bar([])],
        minimum_in_scale_weight_fraction=0.9,
        minimum_best_margin=0.03,
    )

    assert match.scale_root == 2
    assert match.scale_type == ScaleType.MAJOR
    assert match.diagnostics.declared_match_used is False


def test_ambiguous_pitch_set_uses_declared_pitch_set_when_available() -> None:
    match = match_scale(
        [_bar([60, 62, 64, 67, 69, 71], declared_key_fifths=1)],
        [_bar([])],
        minimum_in_scale_weight_fraction=0.9,
        minimum_best_margin=0.03,
    )

    assert match.scale_root == 7
    assert match.scale_type == ScaleType.MAJOR
    assert match.diagnostics.declared_match_used is True
    assert match.diagnostics.ambiguous is False


def test_natural_minor_maps_to_major_pitch_set() -> None:
    match = match_scale(
        [_bar([69, 71, 72, 74, 76, 77, 79])],
        [_bar([])],
        minimum_in_scale_weight_fraction=0.9,
        minimum_best_margin=0.03,
    )

    assert match.scale_root == 0
    assert match.scale_type == ScaleType.MAJOR


def test_harmonic_minor_is_selected_when_raised_seventh_is_present() -> None:
    match = match_scale(
        [_bar([69, 71, 72, 74, 76, 77, 80])],
        [_bar([])],
        minimum_in_scale_weight_fraction=0.9,
        minimum_best_margin=0.03,
    )

    assert match.scale_root == 9
    assert match.scale_type == ScaleType.HARMONIC_MINOR


def test_no_pitches_is_low_confidence_and_marked_explicitly() -> None:
    match = match_scale(
        [_bar([])],
        [_bar([])],
        minimum_in_scale_weight_fraction=0.9,
        minimum_best_margin=0.03,
    )

    assert match.diagnostics.no_pitches is True
    assert match.diagnostics.low_confidence is True


def _bar(midi_pitches: list[int], *, declared_key_fifths: int | None = None) -> ParsedBar:
    return ParsedBar(
        time_numerator=4,
        time_denominator=4,
        declared_key_fifths=declared_key_fifths,
        events=[
            ParsedNote(midi_pitch=midi_pitch, duration=Fraction(1, 4), beat_offset=Fraction(index, 4))
            for index, midi_pitch in enumerate(midi_pitches)
        ],
    )
