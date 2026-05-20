from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from musak_model.data.config import SegmentationConfig
from musak_model.data.converter import PitchDegreeRegisterError
from musak_model.data.pipeline import segment_parsed_score
from musak_model.data.schema import ParsedBar, ParsedNote, ParsedScore, SegmentIneligibilityReason
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType


def _empty_bar() -> ParsedBar:
    return ParsedBar(time_numerator=4, time_denominator=4, key_fifths=0, events=[])


def _note_bar() -> ParsedBar:
    return ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))],
    )


def test_segment_parsed_score_keeps_recoverable_segments_when_feature_extraction_fails(
    duration_vocabulary: DurationVocabulary,
) -> None:
    bad_register_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[ParsedNote(midi_pitch=24, duration=Fraction(1, 4), beat_offset=Fraction(0))],
    )
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[_note_bar(), bad_register_bar, _note_bar()],
        left_hand_bars=[_note_bar(), _note_bar(), _note_bar()],
    )

    segments = segment_parsed_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary,
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert [segment.metadata.eligible_for_training for segment in segments] == [True, False, True]
    assert segments[1].metadata.ineligibility_reasons == {SegmentIneligibilityReason.REGISTER_OUT_OF_RANGE}
    assert segments[0].metadata.difficulty_features is not None
    assert segments[1].metadata.difficulty_features is None


def test_segment_parsed_score_marks_feature_register_errors_ineligible(
    duration_vocabulary: DurationVocabulary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[_note_bar()],
        left_hand_bars=[_note_bar()],
    )

    def fail_feature_extraction(*args, **kwargs):
        raise PitchDegreeRegisterError(24, hand=Hand.RIGHT, octave_offset=-3)

    monkeypatch.setattr("musak_model.data.pipeline.extract_difficulty_features", fail_feature_extraction)

    segments = segment_parsed_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary,
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.REGISTER_OUT_OF_RANGE}
    assert segments[0].metadata.difficulty_features is None


def test_segment_parsed_score_does_not_hide_unexpected_feature_extraction_errors(
    duration_vocabulary: DurationVocabulary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[_note_bar()],
        left_hand_bars=[_note_bar()],
    )

    def fail_feature_extraction(*args, **kwargs):
        raise ValueError("feature extraction bug")

    monkeypatch.setattr("musak_model.data.pipeline.extract_difficulty_features", fail_feature_extraction)

    with pytest.raises(ValueError, match="feature extraction bug"):
        segment_parsed_score(
            score,
            Path("piece.mxl"),
            duration_vocabulary,
            segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
        )
