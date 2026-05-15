from fractions import Fraction
from pathlib import Path

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.config import SegmentationConfig
from musak_model.data.schema import ParsedBar, ParsedNote, ParsedScore, SegmentIneligibilityReason
from musak_model.data.segmenter import segment_score
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType


def _bar(*, time_numerator: int = 4, time_denominator: int = 4, key_fifths: int = 0) -> ParsedBar:
    return ParsedBar(
        time_numerator=time_numerator,
        time_denominator=time_denominator,
        key_fifths=key_fifths,
        events=[],
    )


def _score(*, bars: list[ParsedBar]) -> ParsedScore:
    return ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=bars[0].time_numerator,
        time_denominator=bars[0].time_denominator,
        right_hand_bars=bars,
        left_hand_bars=bars,
    )


def test_segment_metadata_uses_first_bar_time_signature(duration_vocabulary: DurationVocabulary) -> None:
    segments = segment_score(
        _score(bars=[_bar(time_numerator=4, time_denominator=4), _bar(time_numerator=3, time_denominator=4)]),
        Path("piece.mxl"),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert [(segment.time_numerator, segment.time_denominator) for segment in segments] == [(4, 4), (3, 4)]
    assert [segment.metadata.eligible_for_training for segment in segments] == [True, True]


def test_segment_crossing_time_signature_change_is_not_training_eligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    segments = segment_score(
        _score(bars=[_bar(time_numerator=4, time_denominator=4), _bar(time_numerator=3, time_denominator=4)]),
        Path("piece.mxl"),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    assert len(segments) == 1
    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.MIXED_TIME_SIGNATURE}


def test_segment_crossing_key_signature_change_is_not_training_eligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    segments = segment_score(
        _score(bars=[_bar(key_fifths=0), _bar(key_fifths=1)]),
        Path("piece.mxl"),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    assert len(segments) == 1
    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.KEY_SIGNATURE_CHANGE}


def test_tokenization_error_marks_only_affected_segments_ineligible(duration_vocabulary: DurationVocabulary) -> None:
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
        right_hand_bars=[_bar(), bad_register_bar, _bar()],
        left_hand_bars=[_bar(), _bar(), _bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert [segment.metadata.eligible_for_training for segment in segments] == [True, False, True]
    assert segments[1].metadata.ineligibility_reasons == {SegmentIneligibilityReason.TOKENIZATION_ERROR}


def test_ambiguous_simultaneous_durations_mark_segment_ineligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    ambiguous_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[
            ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0)),
            ParsedNote(midi_pitch=64, duration=Fraction(1, 8), beat_offset=Fraction(0)),
        ],
    )
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[ambiguous_bar],
        left_hand_bars=[_bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.AMBIGUOUS_SIMULTANEOUS_DURATION}


def test_cleaned_simultaneous_mixed_durations_are_tokenized_as_chord(
    duration_vocabulary: DurationVocabulary,
) -> None:
    ambiguous_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[
            ParsedNote(midi_pitch=60, duration=Fraction(1, 8), beat_offset=Fraction(0)),
            ParsedNote(midi_pitch=64, duration=Fraction(1, 2), beat_offset=Fraction(0)),
            ParsedNote(midi_pitch=67, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
        ],
    )
    score = clean_parsed_score(
        ParsedScore(
            key_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[ambiguous_bar],
            left_hand_bars=[_bar()],
        )
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is True
    assert segments[0].metadata.ineligibility_reasons == frozenset()


def test_unsupported_quantized_duration_marks_segment_ineligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    unsupported_duration_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 9), beat_offset=Fraction(0))],
    )
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[unsupported_duration_bar],
        left_hand_bars=[_bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.QUANTIZATION_ERROR}


def test_unsupported_rest_gap_marks_segment_ineligible_before_it_can_shift_onsets(
    duration_vocabulary: DurationVocabulary,
) -> None:
    unsupported_rest_gap_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[
            ParsedNote(midi_pitch=60, duration=Fraction(1, 16), beat_offset=Fraction(0)),
            ParsedNote(midi_pitch=60, duration=Fraction(1, 16), beat_offset=Fraction(3, 32)),
        ],
    )
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[unsupported_rest_gap_bar],
        left_hand_bars=[_bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.QUANTIZATION_ERROR}
