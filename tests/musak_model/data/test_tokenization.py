from dataclasses import dataclass
from fractions import Fraction
from typing import List

import pytest

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote, ParsedScore, SegmentMetadata
from musak_model.data.segmenter import _chord_to_tokens, _note_to_token, _tokenize_unified_stream
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken, ScaleType


def _parsed_score() -> ParsedScore:
    return ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
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
    expected_degree: int | None = None
    expected_accidental: int | None = None
    expected_octave_offset: int | None = None
    expected_duration: Fraction | None = None


@dataclass(frozen=True)
class ChordTokenCase:
    midi_pitches: List[int]
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
                key_root=9,
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
        token = _note_to_token(
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
            duration_vocabulary=duration_vocabulary,
        )
        note_tokens = [token for token in tokens if isinstance(token, NoteToken)]

        if case.expected_degree is not None:
            assert any(token.degree == case.expected_degree for token in note_tokens)
        if case.expected_duration is not None:
            expected_duration_id = duration_vocabulary.fraction_to_id(case.expected_duration)
            assert all(token.duration_id == expected_duration_id for token in note_tokens)


def test_chord_to_tokens_preserves_all_chord_pitches(duration_vocabulary: DurationVocabulary) -> None:
    score = _parsed_score()
    chord = ParsedChord(midi_pitches=[60, 64, 67], duration=Fraction(1, 4), beat_offset=Fraction(0, 1))

    tokens = _chord_to_tokens(
        chord,
        score=score,
        hand=Hand.RIGHT,
        duration_vocabulary=duration_vocabulary,
    )

    assert [token.degree for token in tokens if isinstance(token, NoteToken)] == [1, 3, 5]


def test_unified_stream_adds_join_suffixes_for_chord_notes(duration_vocabulary: DurationVocabulary) -> None:
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
                events=[ParsedChord(midi_pitches=[60, 64, 67], duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
        left_hand_bars=[ParsedBar(time_numerator=4, time_denominator=4, key_fifths=0, events=[])],
    )

    tokenized_bars = _tokenize_unified_stream(
        score=score,
        duration_vocabulary=duration_vocabulary,
    )

    assert isinstance(tokenized_bars[0][0], HandToken)
    assert sum(isinstance(token, JoinWithPreviousToken) for token in tokenized_bars[0]) == 2


def test_unified_stream_rejects_overlapping_non_chord_notes(duration_vocabulary: DurationVocabulary) -> None:
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
                    ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0)),
                    ParsedNote(midi_pitch=64, duration=Fraction(1, 4), beat_offset=Fraction(1, 8)),
                ],
            )
        ],
        left_hand_bars=[ParsedBar(time_numerator=4, time_denominator=4, key_fifths=0, events=[])],
    )

    with pytest.raises(ValueError, match="overlapping events"):
        _tokenize_unified_stream(score=score, duration_vocabulary=duration_vocabulary)


def test_unified_stream_rejects_overlap_smaller_than_shortest_supported_duration(
    duration_vocabulary: DurationVocabulary,
) -> None:
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
                    ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0)),
                    ParsedNote(midi_pitch=64, duration=Fraction(1, 4), beat_offset=Fraction(1, 32)),
                ],
            )
        ],
        left_hand_bars=[ParsedBar(time_numerator=4, time_denominator=4, key_fifths=0, events=[])],
    )

    with pytest.raises(ValueError, match="overlapping events"):
        _tokenize_unified_stream(score=score, duration_vocabulary=duration_vocabulary)


def test_cleaned_unified_stream_preserves_notes_after_truncating_overlaps(
    duration_vocabulary: DurationVocabulary,
) -> None:
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
                    ParsedChord(midi_pitches=[60, 64], duration=Fraction(1, 2), beat_offset=Fraction(0)),
                    ParsedNote(midi_pitch=67, duration=Fraction(1, 2), beat_offset=Fraction(1, 4)),
                    ParsedNote(midi_pitch=69, duration=Fraction(1, 2), beat_offset=Fraction(1, 2)),
                ],
            )
        ],
        left_hand_bars=[ParsedBar(time_numerator=4, time_denominator=4, key_fifths=0, events=[])],
    )

    tokenized_bars = _tokenize_unified_stream(
        score=clean_parsed_score(score),
        duration_vocabulary=duration_vocabulary,
    )

    note_tokens = [token for token in tokenized_bars[0] if isinstance(token, NoteToken)]
    assert [duration_vocabulary.id_to_fraction(token.duration_id) for token in note_tokens] == [
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 2),
    ]


class TestTokenizationWithDifferentKeys:
    def test_note_in_c_major_stays_same(self, duration_vocabulary: DurationVocabulary) -> None:
        score = ParsedScore(
            key_root=0,  # C
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[],
            left_hand_bars=[],
        )

        note = ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))
        token = _note_to_token(
            note,
            score=score,
            hand=Hand.RIGHT,
            duration_vocabulary=duration_vocabulary,
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
            expected_duration=Fraction(3, 8),
        ),
        NoteDurationCase(
            midi_pitch=60,
            duration=Fraction(1, 12),
            beat_offset=Fraction(1, 4),
            hand=Hand.RIGHT,
            expected_duration=Fraction(1, 12),
        ),
    ]

    @pytest.mark.parametrize("case", NOTE_DURATION_CASES)
    def test_note_duration_tokenization(self, case: NoteDurationCase, duration_vocabulary: DurationVocabulary) -> None:
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
            duration_vocabulary=duration_vocabulary,
        )

        assert token.duration_id == duration_vocabulary.fraction_to_id(case.expected_duration)
