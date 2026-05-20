from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.evaluation import diagnose_segment
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
)


def test_rest_only_segment_is_empty_and_silent(duration_vocabulary: DurationVocabulary) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            RestToken(duration_id=whole_id),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=whole_id),
        ]
    )

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)

    assert diagnostics.empty_score is True
    assert diagnostics.one_hand_only is False
    assert diagnostics.right_silence_fraction == 1.0
    assert diagnostics.left_silence_fraction == 1.0
    assert diagnostics.both_hands_silence_fraction == 1.0
    assert diagnostics.rest_token_fraction == 0.5


def test_right_hand_only_segment_reports_one_hand_activity(duration_vocabulary: DurationVocabulary) -> None:
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=half_id),
        ]
    )

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)

    assert diagnostics.empty_score is False
    assert diagnostics.one_hand_only is True
    assert diagnostics.right_silence_fraction == 0.5
    assert diagnostics.left_silence_fraction == 1.0
    assert diagnostics.right_only_active_fraction == 0.5
    assert diagnostics.hand_activity_balance == 0.0


def test_simultaneous_hands_report_both_hands_active(duration_vocabulary: DurationVocabulary) -> None:
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=half_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=half_id),
        ]
    )

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)

    assert diagnostics.both_hands_active_fraction == 0.5
    assert diagnostics.both_hands_silence_fraction == 0.5
    assert diagnostics.hand_activity_balance == 1.0


def test_alternating_hands_report_one_sided_activity(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=quarter_id),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ]
    )

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)

    assert diagnostics.right_only_active_fraction == 0.25
    assert diagnostics.left_only_active_fraction == 0.25
    assert diagnostics.both_hands_active_fraction == 0.0
    assert diagnostics.both_hands_silence_fraction == 0.5


def test_chord_notes_count_as_one_onset(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
        ]
    )

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)

    assert diagnostics.right_note_onsets_per_bar == 1.0
    assert diagnostics.note_token_fraction == 0.5


def test_silent_bar_diagnostics_count_edge_and_interior_silent_bars(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            RestToken(duration_id=whole_id),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=whole_id),
            BarToken(),
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=whole_id),
            BarToken(),
            HandToken(hand=Hand.RIGHT),
            RestToken(duration_id=whole_id),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=whole_id),
        ],
        bar_count=3,
    )

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)

    assert diagnostics.silent_bar_count == 2
    assert diagnostics.silent_bar_fraction == 2 / 3
    assert diagnostics.silent_edge_bar_count == 2


def test_hold_token_counts_as_bar_activity(duration_vocabulary: DurationVocabulary) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=whole_id),
            BarToken(),
            HoldToken(duration_id=whole_id),
        ],
        bar_count=2,
    )

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)

    assert diagnostics.silent_bar_count == 0
    assert diagnostics.silent_edge_bar_count == 0


def _segment(tokens: list[Token], *, bar_count: int = 1) -> Segment:
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=bar_count,
            window_start_bar=0,
            source_file=Path("score.mxl"),
        ),
    )
