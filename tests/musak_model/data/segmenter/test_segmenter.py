from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.config import SegmentationConfig, SegmentationMode
from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedNote,
    ParsedRest,
    ParsedScore,
    SegmentIneligibilityReason,
    TieType,
)
from musak_model.data.segmenter.segmenter import segment_score as _segment_score
from musak_model.decoder.piano_roll import segment_to_piano_roll_events
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import HandToken, HoldToken, NoteToken, ScaleType


def _scale_matcher_config() -> ScaleMatcherConfig:
    return ScaleMatcherConfig(
        support_score_margin=0.08,
        selection_score_margin=0.03,
        maximum_unexplained_weight_fraction=0.10,
        maximum_explanation_pitch_class_count=9,
    )


def segment_score(*args: Any, **kwargs: Any):
    return _segment_score(*args, scale_matcher_config=_scale_matcher_config(), **kwargs)


def _bar(*, time_numerator: int = 4, time_denominator: int = 4, key_fifths: int = 0) -> ParsedBar:
    return ParsedBar(
        time_numerator=time_numerator,
        time_denominator=time_denominator,
        key_fifths=key_fifths,
        events=[],
    )


def _rest_bar() -> ParsedBar:
    return ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[ParsedRest(duration=Fraction(1, 1), beat_offset=Fraction(0))],
    )


def _note_bar() -> ParsedBar:
    return ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))],
    )


def _score(*, bars: list[ParsedBar]) -> ParsedScore:
    return ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=bars[0].time_numerator,
        time_denominator=bars[0].time_denominator,
        right_hand_bars=bars,
        left_hand_bars=bars,
    )


def test_segment_metadata_uses_first_bar_time_signature(duration_vocabulary: DurationVocabulary) -> None:
    segments = segment_score(
        _score(
            bars=[
                _note_bar(),
                _note_bar().model_copy(update={"time_numerator": 3, "time_denominator": 4}),
            ]
        ),
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert [(segment.time_numerator, segment.time_denominator) for segment in segments] == [(4, 4), (3, 4)]
    assert [segment.metadata.eligible_for_training for segment in segments] == [True, True]


def test_whole_file_segmentation_creates_one_segment_with_full_bar_count(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = _score(bars=[_note_bar(), _note_bar(), _note_bar(), _note_bar(), _note_bar()])

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1, mode=SegmentationMode.WHOLE_FILE),
    )

    assert len(segments) == 1
    assert segments[0].metadata.bar_count == 5
    assert segments[0].metadata.window_start_bar == 0


def test_segment_crossing_time_signature_change_is_not_training_eligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    segments = segment_score(
        _score(
            bars=[
                _note_bar(),
                _note_bar().model_copy(update={"time_numerator": 3, "time_denominator": 4}),
            ]
        ),
        Path("piece.mxl"),
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
        _score(bars=[_note_bar(), _note_bar().model_copy(update={"declared_key_fifths": 1})]),
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    assert len(segments) == 1
    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.KEY_SIGNATURE_CHANGE}


def test_segment_starting_with_silent_bar_is_not_training_eligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = _score(bars=[_rest_bar(), _note_bar()])

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.SILENT_EDGE_BAR}


def test_segment_ending_with_silent_bar_is_not_training_eligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = _score(bars=[_note_bar(), _rest_bar()])

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.SILENT_EDGE_BAR}


def test_segment_with_interior_silent_bar_remains_training_eligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = _score(bars=[_note_bar(), _rest_bar(), _note_bar()])

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=3, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is True
    assert segments[0].metadata.ineligibility_reasons == frozenset()


def test_cleaned_duplicate_note_events_do_not_make_segment_ineligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    duplicate_note_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[
            ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0)),
            ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0)),
        ],
    )
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[duplicate_note_bar],
        left_hand_bars=[_note_bar()],
    )

    cleaned = clean_parsed_score(score)
    segments = segment_score(
        cleaned,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert cleaned.right_hand_bars[0].events == [
        ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))
    ]
    assert segments[0].metadata.eligible_for_training is True
    assert segments[0].metadata.ineligibility_reasons == frozenset()


def test_register_error_marks_only_affected_segments_ineligible(duration_vocabulary: DurationVocabulary) -> None:
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

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert [segment.metadata.eligible_for_training for segment in segments] == [True, False, True]
    assert segments[1].metadata.ineligibility_reasons == {SegmentIneligibilityReason.REGISTER_OUT_OF_RANGE}


def test_overlapping_events_mark_segment_ineligible(duration_vocabulary: DurationVocabulary) -> None:
    overlapping_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[
            ParsedNote(midi_pitch=60, duration=Fraction(1, 2), beat_offset=Fraction(0)),
            ParsedNote(midi_pitch=64, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
        ],
    )
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[overlapping_bar],
        left_hand_bars=[_bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.OVERLAPPING_EVENTS}


def test_cleaned_overlapping_sequences_tokenize_after_truncating_to_next_onset_and_bar_end(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedChord(midi_pitches=[72, 76], duration=Fraction(1, 2), beat_offset=Fraction(0)),
                    ParsedNote(midi_pitch=79, duration=Fraction(1, 2), beat_offset=Fraction(1, 4)),
                    ParsedNote(midi_pitch=83, duration=Fraction(1, 2), beat_offset=Fraction(1, 2)),
                ],
            )
        ],
        left_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=48, duration=Fraction(1, 1), beat_offset=Fraction(0)),
                    ParsedNote(midi_pitch=55, duration=Fraction(1, 1), beat_offset=Fraction(1, 2)),
                ],
            )
        ],
    )

    segments = segment_score(
        clean_parsed_score(score),
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is True
    events = segment_to_piano_roll_events(segments[0], duration_vocabulary=duration_vocabulary)
    assert [(event.midi_pitch, event.start, event.duration) for event in events] == [
        (72, Fraction(0), Fraction(1, 4)),
        (76, Fraction(0), Fraction(1, 4)),
        (48, Fraction(0), Fraction(1, 2)),
        (79, Fraction(1, 4), Fraction(1, 4)),
        (83, Fraction(1, 2), Fraction(1, 2)),
        (55, Fraction(1, 2), Fraction(1, 2)),
    ]


def test_bar_duration_overflow_marks_segment_ineligible(duration_vocabulary: DurationVocabulary) -> None:
    overflowing_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 1), beat_offset=Fraction(1, 4))],
    )
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[overflowing_bar],
        left_hand_bars=[_bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.BAR_DURATION_OVERFLOW}


def test_tied_note_across_bars_tokenizes_as_hold(duration_vocabulary: DurationVocabulary) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=72, duration=Fraction(1, 1), beat_offset=Fraction(0), tie_type=TieType.START)
                ],
            ),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=72, duration=Fraction(1, 2), beat_offset=Fraction(0), tie_type=TieType.STOP)
                ],
            ),
        ],
        left_hand_bars=[_bar(), _bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    tokens = segments[0].tokens
    assert segments[0].metadata.eligible_for_training is True
    assert sum(isinstance(token, NoteToken) for token in tokens) == 1
    assert sum(isinstance(token, HoldToken) for token in tokens) == 1


def test_window_starting_on_tie_continuation_is_not_training_eligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(
                        midi_pitch=72,
                        duration=Fraction(1, 1),
                        beat_offset=Fraction(0),
                        tie_type=TieType.START,
                    )
                ],
            ),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(
                        midi_pitch=72,
                        duration=Fraction(1, 2),
                        beat_offset=Fraction(0),
                        tie_type=TieType.STOP,
                    )
                ],
            ),
        ],
        left_hand_bars=[_bar(), _bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert [segment.metadata.eligible_for_training for segment in segments] == [True, False]
    assert segments[1].metadata.ineligibility_reasons == {SegmentIneligibilityReason.TIE_CONTINUATION_AT_WINDOW_START}


def test_tied_chord_across_bars_tokenizes_as_single_hold_for_same_pitch_set(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedChord(
                        midi_pitches=[72, 76],
                        duration=Fraction(1, 1),
                        beat_offset=Fraction(0),
                        tie_type=TieType.START,
                    )
                ],
            ),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedChord(
                        midi_pitches=[76, 72],
                        duration=Fraction(1, 2),
                        beat_offset=Fraction(0),
                        tie_type=TieType.STOP,
                    )
                ],
            ),
        ],
        left_hand_bars=[_bar(), _bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    tokens = segments[0].tokens
    assert segments[0].metadata.eligible_for_training is True
    assert sum(isinstance(token, NoteToken) for token in tokens) == 2
    assert sum(isinstance(token, HoldToken) for token in tokens) == 1


def test_multi_bar_ligature_tokenizes_continue_and_stop_as_holds(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(
                        midi_pitch=72,
                        duration=Fraction(1, 1),
                        beat_offset=Fraction(0),
                        tie_type=TieType.START,
                    )
                ],
            ),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(
                        midi_pitch=72,
                        duration=Fraction(1, 1),
                        beat_offset=Fraction(0),
                        tie_type=TieType.CONTINUE,
                    )
                ],
            ),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(
                        midi_pitch=72,
                        duration=Fraction(1, 2),
                        beat_offset=Fraction(0),
                        tie_type=TieType.STOP,
                    )
                ],
            ),
        ],
        left_hand_bars=[_bar(), _bar(), _bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=3, stride_bars=1),
    )

    tokens = segments[0].tokens
    assert segments[0].metadata.eligible_for_training is True
    assert sum(isinstance(token, NoteToken) for token in tokens) == 1
    assert sum(isinstance(token, HoldToken) for token in tokens) == 2


def test_one_hand_can_hold_while_other_hand_plays_regular_notes(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=72, duration=Fraction(1, 1), beat_offset=Fraction(0), tie_type=TieType.START)
                ],
            ),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=72, duration=Fraction(1, 1), beat_offset=Fraction(0), tie_type=TieType.STOP)
                ],
            ),
        ],
        left_hand_bars=[
            _bar(),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0)),
                    ParsedNote(midi_pitch=50, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
                ],
            ),
        ],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    tokens = segments[0].tokens
    assert segments[0].metadata.eligible_for_training is True
    assert any(isinstance(token, HoldToken) for token in tokens)
    assert sum(isinstance(token, NoteToken) for token in tokens) == 3
    assert sum(isinstance(token, HandToken) for token in tokens) >= 3


def test_tie_continuation_without_matching_pitch_marks_segment_ineligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=72, duration=Fraction(1, 1), beat_offset=Fraction(0), tie_type=TieType.START)
                ],
            ),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=74, duration=Fraction(1, 2), beat_offset=Fraction(0), tie_type=TieType.STOP)
                ],
            ),
        ],
        left_hand_bars=[_bar(), _bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=2, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.TIE_MISMATCH}


def test_tie_continuation_without_open_tie_marks_segment_ineligible(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(
                        midi_pitch=72,
                        duration=Fraction(1, 2),
                        beat_offset=Fraction(0),
                        tie_type=TieType.STOP,
                    )
                ],
            )
        ],
        left_hand_bars=[_bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.TIE_MISMATCH}


def test_partial_chord_tie_marks_segment_ineligible(duration_vocabulary: DurationVocabulary) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedChord(
                        midi_pitches=[72, 76],
                        duration=Fraction(1, 2),
                        beat_offset=Fraction(0),
                        tie_type=TieType.PARTIAL,
                    )
                ],
            )
        ],
        left_hand_bars=[_bar()],
    )

    segments = segment_score(
        score,
        Path("piece.mxl"),
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.PARTIAL_CHORD_TIE}


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
        scale_root=0,
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
            scale_root=0,
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
        scale_root=0,
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
        scale_root=0,
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
        duration_vocabulary=duration_vocabulary,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
    )

    assert segments[0].metadata.eligible_for_training is False
    assert segments[0].metadata.ineligibility_reasons == {SegmentIneligibilityReason.QUANTIZATION_ERROR}


def test_unexpected_tokenization_value_error_is_not_swallowed(
    duration_vocabulary: DurationVocabulary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
        left_hand_bars=[_bar()],
    )

    def fail_pitch_conversion(*args, **kwargs):
        raise ValueError("unexpected pitch conversion bug")

    monkeypatch.setattr("musak_model.data.segmenter.bar.pitch_to_degree", fail_pitch_conversion)

    with pytest.raises(ValueError, match="unexpected pitch conversion bug"):
        segment_score(
            score,
            Path("piece.mxl"),
            duration_vocabulary=duration_vocabulary,
            segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        )
