from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.data.schema import Segment, SegmentMetadata
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
)
from notebooks.utils.model_notation import UnsupportedNotationDurationError, segment_to_score_data


def _segment(
    tokens: list[object],
    *,
    time_numerator: int = 4,
    time_denominator: int = 4,
    bar_count: int = 1,
) -> Segment:
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=time_numerator,
            time_denominator=time_denominator,
            bar_count=bar_count,
            window_start_bar=0,
            source_file=Path("generated"),
        ),
    )


def _note(duration_id: int, *, degree: int = 1) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=duration_id)


def test_segment_to_score_data_outputs_two_hand_rows(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id),
            HandToken(hand=Hand.LEFT),
            _note(quarter_id),
        ]
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)

    assert len(score.rows) == 2
    assert score.rows[0][0].clef == "treble"
    assert score.rows[1][0].clef == "bass"


def test_segment_to_score_data_fills_rests(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment([HandToken(hand=Hand.RIGHT), _note(quarter_id)])

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    right_notes = score.rows[0][0].voices[0].notes

    assert right_notes[0].duration == "q"
    assert right_notes[1].duration == "hr"
    assert right_notes[1].dots == 1


def test_segment_to_score_data_groups_joined_chord_notes(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id, degree=1),
            _note(quarter_id, degree=3),
            JoinWithPreviousToken(),
        ]
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    chord = score.rows[0][0].voices[0].notes[0]

    assert chord.duration == "q"
    assert len(chord.keys) == 2


def test_segment_to_score_data_marks_holds_across_barlines_as_ties(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(whole_id),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=whole_id),
            BarToken(),
            HandToken(hand=Hand.RIGHT),
            HoldToken(duration_id=quarter_id),
        ],
        bar_count=2,
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)

    assert score.rows[0][0].voices[0].notes[0].tie_start
    assert score.rows[0][1].voices[0].notes[0].tie_stop


def test_segment_to_score_data_supports_dotted_durations(
    duration_vocabulary: DurationVocabulary,
) -> None:
    dotted_quarter_id = duration_vocabulary.fraction_to_id(Fraction(3, 8))
    segment = _segment(
        [HandToken(hand=Hand.RIGHT), _note(dotted_quarter_id)],
        time_numerator=3,
        time_denominator=8,
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    note = score.rows[0][0].voices[0].notes[0]

    assert note.duration == "q"
    assert note.dots == 1


def test_segment_to_score_data_rejects_unsupported_tuplets(
    duration_vocabulary: DurationVocabulary,
) -> None:
    triplet_eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 12))
    segment = _segment([HandToken(hand=Hand.RIGHT), _note(triplet_eighth_id)])

    with pytest.raises(UnsupportedNotationDurationError, match="not supported"):
        segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
