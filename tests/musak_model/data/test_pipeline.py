from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from musak_model.data.config import (
    SegmentationConfig,
    SegmentationMode,
    load_difficulty_labels,
    load_segmentation_config,
)
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
        scale_root=0,
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
        scale_root=0,
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
        scale_root=0,
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


def test_segment_parsed_score_resolves_relative_path_difficulty_label(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[_note_bar()],
        left_hand_bars=[_note_bar()],
    )

    segments = segment_parsed_score(
        score,
        Path("1/0001.mxl"),
        duration_vocabulary,
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
        difficulty_labels={"1/0001.mxl": 1},
    )

    assert segments[0].metadata.difficulty_level == 1


def test_load_segmentation_config_can_override_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "segmentation.yml"
    config_path.write_text("window_bars: 4\nstride_bars: 2\n", encoding="utf-8")

    config = load_segmentation_config(config_path, mode=SegmentationMode.WHOLE_FILE)

    assert config.window_bars == 4
    assert config.stride_bars == 2
    assert config.mode == SegmentationMode.WHOLE_FILE


def test_load_difficulty_labels_accepts_json_null_labels(tmp_path: Path) -> None:
    labels_path = tmp_path / "difficulty_labels.json"
    labels_path.write_text('{"1/0001.mxl": 1, "unlabeled/example.mxl": null}\n', encoding="utf-8")

    assert load_difficulty_labels(labels_path) == {"1/0001.mxl": 1, "unlabeled/example.mxl": None}
