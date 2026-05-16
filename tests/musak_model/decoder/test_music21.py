from fractions import Fraction
from pathlib import Path

from music21 import chord, note, stream

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.decoder.music21 import segment_to_music21_score
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    ScaleType,
)


def _metadata(*, bar_count: int = 2) -> SegmentMetadata:
    return SegmentMetadata(
        key_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=bar_count,
        window_start_bar=0,
        source_file=Path("score.mxl"),
        difficulty_level=None,
    )


def _part(score: stream.Score, hand: Hand) -> stream.Part:
    for part in score.parts:
        if part.id == hand.value:
            return part

    raise AssertionError(f"{hand.value} hand part not found")


def test_segment_to_music21_score_exports_cross_bar_hold_as_tied_notes(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=whole_id),
            BarToken(),
            HoldToken(duration_id=half_id),
            EndToken(),
        ],
        metadata=_metadata(),
    )

    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    right_notes = list(_part(score, Hand.RIGHT).flatten().notes)

    assert len(right_notes) == 2
    assert [element.pitch.midi for element in right_notes if isinstance(element, note.Note)] == [72, 72]
    assert [element.duration.quarterLength for element in right_notes] == [4, 2]
    assert [element.tie.type for element in right_notes] == ["start", "stop"]


def test_segment_to_music21_score_exports_cross_bar_hold_as_tied_chords(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=whole_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=whole_id),
            JoinWithPreviousToken(),
            BarToken(),
            HoldToken(duration_id=half_id),
            EndToken(),
        ],
        metadata=_metadata(),
    )

    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    right_chords = list(_part(score, Hand.RIGHT).flatten().notes)

    assert len(right_chords) == 2
    assert all(isinstance(element, chord.Chord) for element in right_chords)
    assert [element.duration.quarterLength for element in right_chords] == [4, 2]
    assert [[chord_note.tie.type for chord_note in element.notes] for element in right_chords] == [
        ["start", "start"],
        ["stop", "stop"],
    ]


def test_segment_to_music21_score_leaves_in_bar_notes_untied(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            EndToken(),
        ],
        metadata=_metadata(bar_count=1),
    )

    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    right_notes = list(_part(score, Hand.RIGHT).flatten().notes)

    assert len(right_notes) == 1
    assert right_notes[0].duration.quarterLength == 1
    assert right_notes[0].tie is None
