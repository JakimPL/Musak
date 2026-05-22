from fractions import Fraction

import pytest

from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.data.scale_matcher.matcher import match_scale, match_scale_histogram
from musak_model.data.schema import ParsedBar, ParsedNote
from musak_model.tokens.schema import ScaleType


def test_matches_major_scale_from_duration_weighted_distribution_without_key_signature() -> None:
    match = _match_bars([62, 64, 66, 67, 69, 71, 73])

    assert match.scale_root == 2
    assert match.scale_type == ScaleType.MAJOR
    assert match.diagnostics.in_scale_weight_fraction == 1.0


def test_wrong_declared_key_is_overridden_by_pitch_distribution() -> None:
    match = _match_bars([62, 64, 66, 67, 69, 71, 73], declared_key_fifths=0)

    assert match.scale_root == 2
    assert match.scale_type == ScaleType.MAJOR
    assert match.diagnostics.declared_match_used is False


def test_ambiguous_pitch_set_uses_declared_pitch_set_when_available() -> None:
    match = _match_bars([60, 62, 64, 67, 69, 71], declared_key_fifths=1)

    assert match.scale_root == 7
    assert match.scale_type == ScaleType.MAJOR
    assert match.diagnostics.declared_match_used is True
    assert match.diagnostics.ambiguous is False


def test_natural_minor_maps_to_relative_major_pitch_set() -> None:
    match = _match_bars([69, 71, 72, 74, 76, 77, 79])

    assert match.scale_root == 0
    assert match.scale_type == ScaleType.MAJOR


def test_harmonic_minor_is_selected_when_raised_seventh_is_present() -> None:
    match = _match_bars([69, 71, 72, 74, 76, 77, 80])

    assert match.scale_root == 9
    assert match.scale_type == ScaleType.HARMONIC_MINOR


def test_mixed_minor_sixth_and_seventh_variants_are_explained_by_close_candidates() -> None:
    match = _match_histogram(
        _a_minor_variant_weights(),
        support_score_margin=0.2,
        selection_score_margin=0.2,
        maximum_unexplained_weight_fraction=0.0,
        maximum_explanation_pitch_class_count=9,
    )

    assert match.diagnostics.in_scale_weight_fraction == pytest.approx(22 / 24)
    assert match.diagnostics.explained_out_of_scale_weight_fraction == pytest.approx(2 / 24)
    assert match.diagnostics.unexplained_out_of_scale_weight_fraction == 0.0
    assert match.diagnostics.explanation_pitch_class_count == 9
    assert match.diagnostics.low_confidence is False


def test_declared_key_signature_resolves_mixed_minor_variant_tie_to_relative_major() -> None:
    match = _match_histogram(
        _a_minor_variant_weights(),
        declared_key_fifths=0,
        support_score_margin=0.2,
        selection_score_margin=0.2,
    )

    assert match.scale_root == 0
    assert match.scale_type == ScaleType.MAJOR
    assert match.diagnostics.declared_match_used is True


def test_broad_chromatic_distribution_is_low_confidence_even_when_variants_explain_all_pitches() -> None:
    match = _match_histogram(
        {pitch_class: 1 for pitch_class in range(12)},
        support_score_margin=1.0,
        selection_score_margin=1.0,
        maximum_unexplained_weight_fraction=0.0,
        maximum_explanation_pitch_class_count=9,
    )

    assert match.diagnostics.unexplained_out_of_scale_weight_fraction == 0.0
    assert match.diagnostics.explanation_pitch_class_count == 12
    assert match.diagnostics.low_confidence is True


def test_sparse_histogram_input_is_supported() -> None:
    match = _match_histogram({9: 1, 11: 1, 0: 1, 2: 1, 4: 1, 5: 1, 8: 1})

    assert match.scale_root == 9
    assert match.scale_type == ScaleType.HARMONIC_MINOR
    assert match.diagnostics.observed_pitch_class_count == 7


def test_rejects_invalid_pitch_class_histogram_key() -> None:
    with pytest.raises(ValueError, match="pitch class"):
        _match_histogram({12: 1})


def test_no_pitches_is_low_confidence_and_marked_explicitly() -> None:
    match = _match_bars([])

    assert match.diagnostics.no_pitches is True
    assert match.diagnostics.low_confidence is True


def _match_bars(midi_pitches: list[int], *, declared_key_fifths: int | None = None):
    return match_scale(
        [_bar(midi_pitches, declared_key_fifths=declared_key_fifths)],
        [_bar([])],
        config=_scale_matcher_config(),
    )


def _match_histogram(
    weights: dict[int, int],
    *,
    declared_key_fifths: int | None = None,
    support_score_margin: float = 0.08,
    selection_score_margin: float = 0.03,
    maximum_unexplained_weight_fraction: float = 0.1,
    maximum_explanation_pitch_class_count: int = 9,
):
    return match_scale_histogram(
        {pitch_class: Fraction(weight) for pitch_class, weight in weights.items()},
        declared_key_fifths=declared_key_fifths,
        config=_scale_matcher_config(
            support_score_margin=support_score_margin,
            selection_score_margin=selection_score_margin,
            maximum_unexplained_weight_fraction=maximum_unexplained_weight_fraction,
            maximum_explanation_pitch_class_count=maximum_explanation_pitch_class_count,
        ),
    )


def _scale_matcher_config(
    *,
    support_score_margin: float = 0.08,
    selection_score_margin: float = 0.03,
    maximum_unexplained_weight_fraction: float = 0.1,
    maximum_explanation_pitch_class_count: int = 9,
) -> ScaleMatcherConfig:
    return ScaleMatcherConfig(
        support_score_margin=support_score_margin,
        selection_score_margin=selection_score_margin,
        maximum_unexplained_weight_fraction=maximum_unexplained_weight_fraction,
        maximum_explanation_pitch_class_count=maximum_explanation_pitch_class_count,
    )


def _a_minor_variant_weights() -> dict[int, int]:
    return {
        9: 4,  # A
        11: 4,  # B
        0: 4,  # C
        2: 4,  # D
        4: 4,  # E
        5: 1,  # F
        6: 1,  # F#
        7: 1,  # G
        8: 1,  # G#
    }


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
