from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote, ParsedScore, Segment, SegmentMetadata
from musak_model.decoder import (
    parsed_score_to_piano_roll_events,
    segment_to_music21_score,
    segment_to_piano_roll_events,
    tokens_to_piano_roll_events,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    Hand,
    HandToken,
    JoinWithPreviousToken,
    NoteToken,
    ScaleType,
)


def _metadata() -> SegmentMetadata:
    return SegmentMetadata(
        key_root=0,
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


def test_segment_to_music21_score_groups_same_onset_notes_as_chord(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
        ],
        metadata=_metadata(),
    )

    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    right_notes = list(score.parts[0].flatten().notes)

    assert len(right_notes) == 1
    assert len(right_notes[0].pitches) == 2


def test_segment_to_piano_roll_events_uses_legacy_hand_tokens_when_unified_stream_is_empty(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        right_hand_tokens=[NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id)],
        left_hand_tokens=[NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id)],
        metadata=_metadata(),
    )

    events = segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary)

    assert {event.hand for event in events} == {Hand.RIGHT, Hand.LEFT}


def test_parsed_score_to_piano_roll_events_does_not_require_tokenized_segment() -> None:
    score = ParsedScore(
        key_root=0,
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
                        midi_pitches=[60, 64],
                        duration=Fraction(1, 4),
                        beat_offset=Fraction(0),
                    )
                ],
            )
        ],
        left_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(
                        midi_pitch=48,
                        duration=Fraction(1, 2),
                        beat_offset=Fraction(1, 4),
                    )
                ],
            )
        ],
    )

    events = parsed_score_to_piano_roll_events(score)

    assert [(event.hand, event.midi_pitch, event.start, event.duration) for event in events] == [
        (Hand.RIGHT, 60, Fraction(0), Fraction(1, 4)),
        (Hand.RIGHT, 64, Fraction(0), Fraction(1, 4)),
        (Hand.LEFT, 48, Fraction(1, 4), Fraction(1, 2)),
    ]
