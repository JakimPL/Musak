from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.data.schema import Segment, SegmentMetadata, SpellingContextSource, TokenizationContext
from musak_model.decoder.notation import (
    UnsupportedNotationDurationError,
    segment_to_notation_events,
    segment_to_score_data,
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
)


def _segment(
    tokens: list[object],
    *,
    scale_root: int = 0,
    scale_type: ScaleType = ScaleType.MAJOR,
    tokenization_context: TokenizationContext | None = None,
    time_numerator: int = 4,
    time_denominator: int = 4,
    bar_count: int = 1,
) -> Segment:
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=scale_root,
            scale_type=scale_type,
            tokenization_context=tokenization_context,
            time_numerator=time_numerator,
            time_denominator=time_denominator,
            bar_count=bar_count,
            window_start_bar=0,
            source_file=Path("generated"),
        ),
    )


def _note(duration_id: int, *, degree: int = 1, accidental: int = 0) -> NoteToken:
    return NoteToken(degree=degree, accidental=accidental, octave_offset=0, duration_id=duration_id)


def _tokenization_context(
    *,
    pitch_set_scale_root: int,
    pitch_set_scale_type: ScaleType,
    declared_key_fifths: int | None,
    spelling_key_fifths: int,
    spelling_context_source: SpellingContextSource,
) -> TokenizationContext:
    return TokenizationContext(
        pitch_set_scale_root=pitch_set_scale_root,
        pitch_set_scale_type=pitch_set_scale_type,
        declared_key_fifths=declared_key_fifths,
        spelling_key_fifths=spelling_key_fifths,
        spelling_context_source=spelling_context_source,
    )


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


def test_segment_to_score_data_outputs_grand_staff_pairs_when_requested(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id),
            HandToken(hand=Hand.LEFT),
            _note(quarter_id),
            BarToken(),
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id),
            HandToken(hand=Hand.LEFT),
            _note(quarter_id),
        ],
        bar_count=2,
    )

    score = segment_to_score_data(
        segment,
        duration_vocabulary=duration_vocabulary,
        layout="grand_staff",
    )

    assert score.layout == "grand_staff"
    assert len(score.rows) == 1
    assert [stave.clef for stave in score.rows[0]] == ["treble", "bass", "treble", "bass"]


def test_segment_to_score_data_sets_first_measure_key_and_time_signatures(
    duration_vocabulary: DurationVocabulary,
) -> None:
    segment = _segment(
        [],
        scale_root=2,
        scale_type=ScaleType.MAJOR,
        time_numerator=3,
        time_denominator=4,
        bar_count=2,
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)

    assert score.rows[0][0].key_signature == "D"
    assert score.rows[0][0].time_signature == (3, 4)
    assert score.rows[0][1].key_signature is None
    assert score.rows[0][1].time_signature is None
    assert score.rows[1][0].key_signature == "D"
    assert score.rows[1][0].time_signature == (3, 4)


def test_segment_to_score_data_uses_parent_major_key_signature_for_minor_scales(
    duration_vocabulary: DurationVocabulary,
) -> None:
    segment = _segment([], scale_root=9, scale_type=ScaleType.HARMONIC_MINOR)

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)

    assert score.rows[0][0].key_signature == "C"


def test_segment_to_score_data_uses_spelling_context_key_signature(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [HandToken(hand=Hand.RIGHT), _note(quarter_id, degree=4)],
        scale_root=5,
        scale_type=ScaleType.MAJOR,
        tokenization_context=_tokenization_context(
            pitch_set_scale_root=5,
            pitch_set_scale_type=ScaleType.MAJOR,
            declared_key_fifths=0,
            spelling_key_fifths=0,
            spelling_context_source=SpellingContextSource.DECLARED_KEY_SIGNATURE,
        ),
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    note = score.rows[0][0].voices[0].notes[0]

    assert score.rows[0][0].key_signature == "C"
    assert note.keys == ["bb/5"]
    assert note.accidentals == ["b"]


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


def test_joined_chord_keeps_following_note_at_chord_end(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id, degree=1),
            _note(quarter_id, degree=3),
            JoinWithPreviousToken(),
            _note(quarter_id, degree=5),
        ]
    )

    events = segment_to_notation_events(segment, duration_vocabulary=duration_vocabulary)

    assert [event.start for event in events] == [Fraction(0), Fraction(0), Fraction(1, 4)]


def test_segment_to_score_data_does_not_insert_rest_after_joined_chord(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id, degree=1),
            _note(quarter_id, degree=3),
            JoinWithPreviousToken(),
            _note(quarter_id, degree=5),
        ]
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    right_notes = score.rows[0][0].voices[0].notes

    assert [(note.duration, note.keys) for note in right_notes] == [
        ("q", ["c/5", "e/5"]),
        ("q", ["g/5"]),
        ("hr", ["b/4"]),
    ]


def test_segment_to_score_data_preserves_token_accidental_spelling(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id, degree=1, accidental=1),
            _note(quarter_id, degree=2, accidental=-1),
        ]
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    notes = score.rows[0][0].voices[0].notes

    assert notes[0].keys == ["c#/5"]
    assert notes[0].accidentals == ["#"]
    assert notes[1].keys == ["db/5"]
    assert notes[1].accidentals == ["b"]


def test_segment_to_score_data_suppresses_accidentals_declared_by_key_signature(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [HandToken(hand=Hand.RIGHT), _note(quarter_id, degree=3)],
        scale_root=2,
        scale_type=ScaleType.MAJOR,
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    note = score.rows[0][0].voices[0].notes[0]

    assert score.rows[0][0].key_signature == "D"
    assert note.keys == ["f#/5"]
    assert note.accidentals == [None]


def test_segment_to_score_data_spells_flat_key_signature_notes_by_key_signature(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [HandToken(hand=Hand.RIGHT), _note(quarter_id, degree=1)],
        scale_root=1,
        scale_type=ScaleType.MAJOR,
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    note = score.rows[0][0].voices[0].notes[0]

    assert score.rows[0][0].key_signature == "Db"
    assert note.keys == ["db/5"]
    assert note.accidentals == [None]


def test_segment_to_score_data_marks_natural_that_cancels_key_signature(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [HandToken(hand=Hand.RIGHT), _note(quarter_id, degree=3, accidental=-1)],
        scale_root=2,
        scale_type=ScaleType.MAJOR,
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    note = score.rows[0][0].voices[0].notes[0]

    assert score.rows[0][0].key_signature == "D"
    assert note.keys == ["f/5"]
    assert note.accidentals == ["n"]


def test_segment_to_score_data_uses_token_bars_when_metadata_bar_count_is_lower(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id),
            BarToken(),
            HandToken(hand=Hand.RIGHT),
            _note(quarter_id),
            BarToken(),
        ],
        bar_count=0,
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)

    assert len(score.rows[0]) == 2
    assert len(score.rows[1]) == 2


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


def test_segment_to_score_data_splits_composite_rests(
    duration_vocabulary: DurationVocabulary,
) -> None:
    segment = _segment(
        [],
        time_numerator=9,
        time_denominator=16,
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    right_notes = score.rows[0][0].voices[0].notes

    assert [(note.duration, note.dots) for note in right_notes] == [("hr", 0), ("16r", 0)]


def test_segment_to_score_data_splits_composite_notes_with_ties(
    duration_vocabulary: DurationVocabulary,
) -> None:
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    sixteenth_id = duration_vocabulary.fraction_to_id(Fraction(1, 16))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(half_id),
            HoldToken(duration_id=sixteenth_id),
        ]
    )

    score = segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
    right_notes = score.rows[0][0].voices[0].notes

    assert [(note.duration, note.dots) for note in right_notes[:2]] == [("h", 0), ("16", 0)]
    assert right_notes[0].tie_start
    assert right_notes[1].tie_stop


def test_segment_to_score_data_rejects_unsupported_tuplets(
    duration_vocabulary: DurationVocabulary,
) -> None:
    triplet_eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 12))
    segment = _segment([HandToken(hand=Hand.RIGHT), _note(triplet_eighth_id)])

    with pytest.raises(UnsupportedNotationDurationError, match="not supported"):
        segment_to_score_data(segment, duration_vocabulary=duration_vocabulary)
