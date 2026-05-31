from fractions import Fraction
from pathlib import Path

from music21 import chord, key, note, stream

from musak_model.data.schema import Segment, SegmentMetadata, SpellingContextSource, TokenizationContext
from musak_model.data.tokenization_context import tokenization_context_from_scale
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


def _metadata(
    *,
    bar_count: int = 2,
    scale_root: int = 0,
    scale_type: ScaleType = ScaleType.MAJOR,
) -> SegmentMetadata:
    return _metadata_with_tokenization_context(
        bar_count=bar_count,
        scale_root=scale_root,
        scale_type=scale_type,
        tokenization_context=tokenization_context_from_scale(scale_root=scale_root, scale_type=scale_type),
    )


def _metadata_with_tokenization_context(
    *,
    bar_count: int,
    scale_root: int,
    scale_type: ScaleType,
    tokenization_context: TokenizationContext,
) -> SegmentMetadata:
    return SegmentMetadata(
        scale_root=scale_root,
        scale_type=scale_type,
        tokenization_context=tokenization_context,
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


def _key_signature(score: stream.Score, hand: Hand) -> key.KeySignature:
    key_signatures = list(_part(score, hand).flatten().getElementsByClass(key.KeySignature))
    assert len(key_signatures) == 1
    return key_signatures[0]


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


def test_segment_to_music21_score_exports_spelling_context_key_signature(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[HandToken(hand=Hand.RIGHT), NoteToken(degree=7, accidental=0, octave_offset=0, duration_id=quarter_id)],
        metadata=_metadata(bar_count=1, scale_root=9, scale_type=ScaleType.HARMONIC_MINOR),
    )

    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    right_notes = list(_part(score, Hand.RIGHT).flatten().notes)

    assert _key_signature(score, Hand.RIGHT).sharps == 0
    assert right_notes[0].pitch.nameWithOctave == "G#5"


def test_segment_to_music21_score_preserves_modal_spelling_context(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    tokenization_context = TokenizationContext(
        pitch_set_scale_root=5,
        pitch_set_scale_type=ScaleType.MAJOR,
        declared_key_fifths=0,
        spelling_key_fifths=0,
        spelling_context_source=SpellingContextSource.DECLARED_KEY_SIGNATURE,
    )
    segment = Segment(
        tokens=[HandToken(hand=Hand.RIGHT), NoteToken(degree=4, accidental=0, octave_offset=0, duration_id=quarter_id)],
        metadata=_metadata_with_tokenization_context(
            bar_count=1,
            scale_root=5,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context,
        ),
    )

    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    right_notes = list(_part(score, Hand.RIGHT).flatten().notes)

    assert _key_signature(score, Hand.RIGHT).sharps == 0
    assert right_notes[0].pitch.nameWithOctave == "B-5"


def test_segment_to_music21_score_groups_same_onset_notes_as_chord(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
        ],
        metadata=_metadata(bar_count=1),
    )

    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    right_notes = list(_part(score, Hand.RIGHT).flatten().notes)

    assert len(right_notes) == 1
    assert len(right_notes[0].pitches) == 2
