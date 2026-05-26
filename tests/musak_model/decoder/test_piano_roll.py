from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote, ParsedScore, Segment, SegmentMetadata
from musak_model.decoder.piano_roll import (
    parsed_score_to_piano_roll_events,
    segment_to_piano_roll_events,
    tokens_to_piano_roll_events,
)
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
    StartToken,
)


def _metadata() -> SegmentMetadata:
    return SegmentMetadata(
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=1,
        window_start_bar=0,
        source_file=Path("score.mxl"),
        difficulty_level=None,
    )


def test_tokens_to_piano_roll_events_decodes_pitch_and_time(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    events = tokens_to_piano_roll_events(
        [
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            BarToken(),
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=_metadata(),
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )

    assert [(event.midi_pitch, event.start, event.duration) for event in events] == [
        (72, Fraction(0), Fraction(1, 4)),
        (79, Fraction(1), Fraction(1, 4)),
    ]


def test_tokens_to_piano_roll_events_ignores_start_token(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    events = tokens_to_piano_roll_events(
        [
            StartToken(),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=_metadata(),
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )

    assert [(event.midi_pitch, event.start) for event in events] == [(72, Fraction(0))]


def test_join_with_previous_token_keeps_chord_notes_at_same_start(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    events = tokens_to_piano_roll_events(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
        ],
        metadata=_metadata(),
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )

    assert [event.start for event in events] == [Fraction(0), Fraction(0), Fraction(0)]
    assert [event.midi_pitch for event in events] == [72, 76, 79]


def test_unified_two_hand_decoding_uses_independent_hand_cursors(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    events = tokens_to_piano_roll_events(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=quarter_id),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=_metadata(),
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )

    assert [(event.hand, event.start) for event in events] == [
        (Hand.RIGHT, Fraction(0)),
        (Hand.LEFT, Fraction(1, 4)),
    ]


def test_hold_token_extends_previous_same_hand_note_without_new_attack(
    duration_vocabulary: DurationVocabulary,
) -> None:
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    events = tokens_to_piano_roll_events(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=half_id),
            BarToken(),
            HoldToken(duration_id=half_id),
        ],
        metadata=_metadata(),
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )

    assert len(events) == 1
    assert events[0].hand == Hand.RIGHT
    assert events[0].start == Fraction(0)
    assert events[0].duration == Fraction(1, 1)


def test_hold_token_extends_same_hand_chord_notes(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    events = tokens_to_piano_roll_events(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
            BarToken(),
            HoldToken(duration_id=quarter_id),
        ],
        metadata=_metadata(),
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )

    assert [event.start for event in events] == [Fraction(0), Fraction(0)]
    assert [event.duration for event in events] == [Fraction(1, 2), Fraction(1, 2)]
    assert [event.midi_pitch for event in events] == [72, 76]


def test_hold_token_is_scoped_to_active_hand_while_other_hand_plays(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    events = tokens_to_piano_roll_events(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=half_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=half_id),
            BarToken(),
            HandToken(hand=Hand.RIGHT),
            HoldToken(duration_id=half_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=_metadata(),
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )

    right_events = [event for event in events if event.hand == Hand.RIGHT]
    left_events = [event for event in events if event.hand == Hand.LEFT]

    assert [(event.start, event.duration) for event in right_events] == [(Fraction(0), Fraction(1, 1))]
    assert [(event.start, event.duration) for event in left_events] == [
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 1), Fraction(1, 4)),
        (Fraction(5, 4), Fraction(1, 4)),
    ]


def test_hold_token_rejects_missing_same_hand_note(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))

    with pytest.raises(ValueError, match="left hand"):
        tokens_to_piano_roll_events(
            [
                HandToken(hand=Hand.RIGHT),
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
                HandToken(hand=Hand.LEFT),
                HoldToken(duration_id=quarter_id),
            ],
            metadata=_metadata(),
            duration_vocabulary=duration_vocabulary,
            default_hand=Hand.RIGHT,
        )


def test_segment_to_piano_roll_events_decodes_canonical_unified_tokens(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=_metadata(),
    )

    events = segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary)

    assert {event.hand for event in events} == {Hand.RIGHT, Hand.LEFT}


def test_parsed_score_to_piano_roll_events_does_not_require_tokenized_segment() -> None:
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
                events=[ParsedChord(midi_pitches=[60, 64], duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
        left_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=48, duration=Fraction(1, 2), beat_offset=Fraction(1, 4))],
            )
        ],
    )

    events = parsed_score_to_piano_roll_events(score)

    assert [(event.hand, event.midi_pitch, event.start, event.duration) for event in events] == [
        (Hand.RIGHT, 60, Fraction(0), Fraction(1, 4)),
        (Hand.RIGHT, 64, Fraction(0), Fraction(1, 4)),
        (Hand.LEFT, 48, Fraction(1, 4), Fraction(1, 2)),
    ]
