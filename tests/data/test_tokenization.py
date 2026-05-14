from dataclasses import dataclass
from fractions import Fraction
from typing import List

import pytest

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote, ParsedScore, SegmentMetadata
from musak_model.data.segmenter import _chord_to_tokens, _note_to_token, _tokenize_unified_stream
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration_vocabulary import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken, ScaleType


def _vocabulary() -> DurationVocabulary:
    return DurationVocabulary(TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1))


def _parsed_score() -> ParsedScore:
    return ParsedScore(
        key_root=0,
        key_fifths=0,
        mode="major",
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[],
        left_hand_bars=[],
    )


def test_parsed_events_accept_zero_beat_offset() -> None:
    assert ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0)).beat_offset == 0


@dataclass(frozen=True)
class NoteTokenCase:
    note: ParsedNote
    score: ParsedScore
    hand: Hand
    scale_type: ScaleType
    expected_degree: int | None = None
    expected_accidental: int | None = None
    expected_octave_offset: int | None = None
    expected_duration_id: int | None = None


@dataclass(frozen=True)
class ChordTokenCase:
    midi_pitches: List[int]
    duration: Fraction
    beat_offset: Fraction
    hand: Hand
    scale_type: ScaleType
    expected_degree: int | None = None
    expected_duration_id: int | None = None


@dataclass(frozen=True)
class NoteDurationCase:
    midi_pitch: int
    duration: Fraction
    beat_offset: Fraction
    hand: Hand
    scale_type: ScaleType
    expected_duration_id: int


class TestNoteToToken:
    CASES = [
        NoteTokenCase(
            note=ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            score=_parsed_score(),
            hand=Hand.RIGHT,
            scale_type=ScaleType.MAJOR,
            expected_degree=1,
            expected_accidental=0,
            expected_octave_offset=0,
            expected_duration_id=DurationVocabulary(
                TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1)
            ).fraction_to_id(Fraction(1, 4)),
        ),
        NoteTokenCase(
            note=ParsedNote(midi_pitch=61, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            score=_parsed_score(),
            hand=Hand.RIGHT,
            scale_type=ScaleType.MAJOR,
            expected_degree=1,
            expected_accidental=1,
            expected_octave_offset=0,
            expected_duration_id=DurationVocabulary(
                TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1)
            ).fraction_to_id(Fraction(1, 4)),
        ),
        NoteTokenCase(
            note=ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            score=_parsed_score(),
            hand=Hand.LEFT,
            scale_type=ScaleType.MAJOR,
            expected_degree=1,
            expected_accidental=0,
            expected_octave_offset=1,
            expected_duration_id=DurationVocabulary(
                TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1)
            ).fraction_to_id(Fraction(1, 4)),
        ),
        NoteTokenCase(
            note=ParsedNote(midi_pitch=68, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            score=ParsedScore(
                key_root=9,
                key_fifths=0,
                mode="minor",
                time_numerator=4,
                time_denominator=4,
                right_hand_bars=[],
                left_hand_bars=[],
            ),
            hand=Hand.RIGHT,
            scale_type=ScaleType.HARMONIC_MINOR,
            expected_degree=7,
            expected_accidental=0,
            expected_octave_offset=0,
            expected_duration_id=DurationVocabulary(
                TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1)
            ).fraction_to_id(Fraction(1, 4)),
        ),
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_note_to_token(self, case: NoteTokenCase) -> None:
        vocab = _vocabulary()
        token = _note_to_token(
            case.note, score=case.score, hand=case.hand, scale_type=case.scale_type, duration_vocabulary=vocab
        )

        assert isinstance(token, NoteToken)
        assert token.degree == case.expected_degree
        assert token.accidental == case.expected_accidental
        assert token.octave_offset == case.expected_octave_offset
        assert token.duration_id == case.expected_duration_id


class TestChordToTokens:
    CHORD_CASES = [
        ChordTokenCase(
            midi_pitches=[60, 64, 67],
            duration=Fraction(1, 4),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            scale_type=ScaleType.MAJOR,
            expected_degree=5,
        ),
        ChordTokenCase(
            midi_pitches=[60, 64, 67],
            duration=Fraction(1, 4),
            beat_offset=Fraction(1, 4),
            hand=Hand.LEFT,
            scale_type=ScaleType.MAJOR,
            expected_degree=1,
        ),
        ChordTokenCase(
            midi_pitches=[60, 64, 67],
            duration=Fraction(3, 8),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            scale_type=ScaleType.MAJOR,
            expected_duration_id=DurationVocabulary(
                TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1)
            ).fraction_to_id(Fraction(3, 8)),
        ),
    ]

    @pytest.mark.parametrize("case", CHORD_CASES)
    def test_chord_to_tokens(self, case: ChordTokenCase) -> None:
        vocab = _vocabulary()
        score = _parsed_score()

        chord = ParsedChord(
            midi_pitches=case.midi_pitches,
            duration=case.duration,
            beat_offset=case.beat_offset,
        )
        tokens = _chord_to_tokens(
            chord,
            score=score,
            hand=case.hand,
            scale_type=case.scale_type,
            duration_vocabulary=vocab,
        )
        note_tokens = [token for token in tokens if isinstance(token, NoteToken)]

        if case.expected_degree is not None:
            assert any(token.degree == case.expected_degree for token in note_tokens)
        if case.expected_duration_id is not None:
            assert all(token.duration_id == case.expected_duration_id for token in note_tokens)


def test_chord_to_tokens_preserves_all_chord_pitches() -> None:
    vocab = _vocabulary()
    score = _parsed_score()
    chord = ParsedChord(midi_pitches=[60, 64, 67], duration=Fraction(1, 4), beat_offset=Fraction(0, 1))

    tokens = _chord_to_tokens(
        chord,
        score=score,
        hand=Hand.RIGHT,
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=vocab,
    )

    assert [token.degree for token in tokens if isinstance(token, NoteToken)] == [1, 3, 5]


def test_unified_stream_adds_join_suffixes_for_chord_notes() -> None:
    vocab = _vocabulary()
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        mode="major",
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(events=[ParsedChord(midi_pitches=[60, 64, 67], duration=Fraction(1, 4), beat_offset=Fraction(0))])
        ],
        left_hand_bars=[ParsedBar(events=[])],
    )

    tokenized_bars = _tokenize_unified_stream(
        score=score,
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=vocab,
    )

    assert isinstance(tokenized_bars[0][0], HandToken)
    assert sum(isinstance(token, JoinWithPreviousToken) for token in tokenized_bars[0]) == 2


def test_unified_stream_rejects_overlapping_non_chord_notes() -> None:
    vocab = _vocabulary()
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        mode="major",
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                events=[
                    ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0)),
                    ParsedNote(midi_pitch=64, duration=Fraction(1, 4), beat_offset=Fraction(1, 8)),
                ]
            )
        ],
        left_hand_bars=[ParsedBar(events=[])],
    )

    with pytest.raises(ValueError, match="overlapping events"):
        _tokenize_unified_stream(score=score, scale_type=ScaleType.MAJOR, duration_vocabulary=vocab)


class TestTokenizationWithDifferentKeys:
    def test_note_in_c_major_stays_same(self) -> None:
        vocab = _vocabulary()
        score = ParsedScore(
            key_root=0,  # C
            key_fifths=0,
            mode="major",
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[],
            left_hand_bars=[],
        )

        note = ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))
        token = _note_to_token(
            note, score=score, hand=Hand.RIGHT, scale_type=ScaleType.MAJOR, duration_vocabulary=vocab
        )

        assert token.degree == 1
        assert token.accidental == 0


class TestTokenizationEdgeCases:
    NOTE_DURATION_CASES = [
        NoteDurationCase(
            midi_pitch=60,
            duration=Fraction(3, 8),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            scale_type=ScaleType.MAJOR,
            expected_duration_id=DurationVocabulary(
                TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1)
            ).fraction_to_id(Fraction(3, 8)),
        ),
        NoteDurationCase(
            midi_pitch=60,
            duration=Fraction(1, 12),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            scale_type=ScaleType.MAJOR,
            expected_duration_id=DurationVocabulary(
                TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1)
            ).fraction_to_id(Fraction(1, 12)),
        ),
    ]

    @pytest.mark.parametrize("case", NOTE_DURATION_CASES)
    def test_note_duration_tokenization(self, case: NoteDurationCase) -> None:
        vocab = _vocabulary()
        score = _parsed_score()

        note = ParsedNote(
            midi_pitch=case.midi_pitch,
            duration=case.duration,
            beat_offset=case.beat_offset,
        )
        token = _note_to_token(
            note,
            score=score,
            hand=case.hand,
            scale_type=case.scale_type,
            duration_vocabulary=vocab,
        )

        assert token.duration_id == case.expected_duration_id
