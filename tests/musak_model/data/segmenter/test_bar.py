from dataclasses import dataclass
from fractions import Fraction

import pytest

from musak_model.data.schema import ParsedChord, ParsedNote, ParsedScore
from musak_model.data.segmenter.bar import chord_to_tokens, note_to_token
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, NoteToken, ScaleType


def _parsed_score() -> ParsedScore:
    return ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[],
        left_hand_bars=[],
    )


@dataclass(frozen=True)
class NoteTokenCase:
    note: ParsedNote
    score: ParsedScore
    hand: Hand
    expected_degree: int | None = None
    expected_accidental: int | None = None
    expected_octave_offset: int | None = None
    expected_duration: Fraction | None = None


@dataclass(frozen=True)
class ChordTokenCase:
    midi_pitches: list[int]
    duration: Fraction
    beat_offset: Fraction
    hand: Hand
    expected_degree: int | None = None
    expected_duration: Fraction | None = None


@dataclass(frozen=True)
class NoteDurationCase:
    midi_pitch: int
    duration: Fraction
    beat_offset: Fraction
    hand: Hand
    expected_duration: Fraction


class TestNoteToToken:
    CASES = [
        NoteTokenCase(
            note=ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            score=_parsed_score(),
            hand=Hand.RIGHT,
            expected_degree=1,
            expected_accidental=0,
            expected_octave_offset=-1,
            expected_duration=Fraction(1, 4),
        ),
        NoteTokenCase(
            note=ParsedNote(midi_pitch=61, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            score=_parsed_score(),
            hand=Hand.RIGHT,
            expected_degree=1,
            expected_accidental=1,
            expected_octave_offset=-1,
            expected_duration=Fraction(1, 4),
        ),
        NoteTokenCase(
            note=ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            score=_parsed_score(),
            hand=Hand.LEFT,
            expected_degree=1,
            expected_accidental=0,
            expected_octave_offset=1,
            expected_duration=Fraction(1, 4),
        ),
        NoteTokenCase(
            note=ParsedNote(midi_pitch=68, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            score=ParsedScore(
                scale_root=9,
                key_fifths=0,
                scale_type=ScaleType.HARMONIC_MINOR,
                time_numerator=4,
                time_denominator=4,
                right_hand_bars=[],
                left_hand_bars=[],
            ),
            hand=Hand.RIGHT,
            expected_degree=7,
            expected_accidental=0,
            expected_octave_offset=-1,
            expected_duration=Fraction(1, 4),
        ),
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_note_to_token(self, case: NoteTokenCase, duration_vocabulary: DurationVocabulary) -> None:
        token = note_to_token(
            case.note,
            score=case.score,
            hand=case.hand,
            duration_vocabulary=duration_vocabulary,
        )

        assert isinstance(token, NoteToken)
        assert token.degree == case.expected_degree
        assert token.accidental == case.expected_accidental
        assert token.octave_offset == case.expected_octave_offset
        assert token.duration_id == duration_vocabulary.fraction_to_id(case.expected_duration)


class TestChordToTokens:
    CHORD_CASES = [
        ChordTokenCase(
            midi_pitches=[60, 64, 67],
            duration=Fraction(1, 4),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            expected_degree=5,
        ),
        ChordTokenCase(
            midi_pitches=[60, 64, 67],
            duration=Fraction(1, 4),
            beat_offset=Fraction(1, 4),
            hand=Hand.LEFT,
            expected_degree=1,
        ),
        ChordTokenCase(
            midi_pitches=[60, 64, 67],
            duration=Fraction(3, 8),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            expected_duration=Fraction(3, 8),
        ),
    ]

    @pytest.mark.parametrize("case", CHORD_CASES)
    def test_chord_to_tokens(self, case: ChordTokenCase, duration_vocabulary: DurationVocabulary) -> None:
        chord = ParsedChord(
            midi_pitches=case.midi_pitches,
            duration=case.duration,
            beat_offset=case.beat_offset,
        )
        tokens = chord_to_tokens(
            chord,
            score=_parsed_score(),
            hand=case.hand,
            duration_vocabulary=duration_vocabulary,
        )
        note_tokens = [token for token in tokens if isinstance(token, NoteToken)]

        if case.expected_degree is not None:
            assert any(token.degree == case.expected_degree for token in note_tokens)
        if case.expected_duration is not None:
            expected_duration_id = duration_vocabulary.fraction_to_id(case.expected_duration)
            assert all(token.duration_id == expected_duration_id for token in note_tokens)


def test_chord_to_tokens_preserves_all_chord_pitches(duration_vocabulary: DurationVocabulary) -> None:
    chord = ParsedChord(midi_pitches=[60, 64, 67], duration=Fraction(1, 4), beat_offset=Fraction(0))

    tokens = chord_to_tokens(
        chord,
        score=_parsed_score(),
        hand=Hand.RIGHT,
        duration_vocabulary=duration_vocabulary,
    )

    assert [token.degree for token in tokens if isinstance(token, NoteToken)] == [1, 3, 5]


def test_note_in_c_major_stays_same(duration_vocabulary: DurationVocabulary) -> None:
    note = ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))
    token = note_to_token(
        note,
        score=_parsed_score(),
        hand=Hand.RIGHT,
        duration_vocabulary=duration_vocabulary,
    )

    assert token.degree == 1
    assert token.accidental == 0


@pytest.mark.parametrize(
    "case",
    [
        NoteDurationCase(
            midi_pitch=60,
            duration=Fraction(3, 8),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            expected_duration=Fraction(3, 8),
        ),
        NoteDurationCase(
            midi_pitch=60,
            duration=Fraction(1, 12),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            expected_duration=Fraction(1, 12),
        ),
    ],
)
def test_note_duration_tokenization(case: NoteDurationCase, duration_vocabulary: DurationVocabulary) -> None:
    note = ParsedNote(
        midi_pitch=case.midi_pitch,
        duration=case.duration,
        beat_offset=case.beat_offset,
    )
    token = note_to_token(
        note,
        score=_parsed_score(),
        hand=case.hand,
        duration_vocabulary=duration_vocabulary,
    )

    assert token.duration_id == duration_vocabulary.fraction_to_id(case.expected_duration)
