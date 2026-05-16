from fractions import Fraction

import pytest

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote, ParsedScore
from musak_model.data.segmenter import _chord_to_tokens, _tokenize_unified_stream
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken, ScaleType


def _score(*, right_hand_bars: list[ParsedBar], left_hand_bars: list[ParsedBar]) -> ParsedScore:
    return ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=right_hand_bars,
        left_hand_bars=left_hand_bars,
    )


def test_parsed_notes_accept_zero_beat_offset() -> None:
    note = ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))

    assert note.beat_offset == 0


def test_chord_tokenization_preserves_all_chord_pitches(duration_vocabulary: DurationVocabulary) -> None:
    chord = ParsedChord(midi_pitches=[60, 64, 67], duration=Fraction(1, 4), beat_offset=Fraction(0))

    tokens = _chord_to_tokens(
        chord,
        score=_score(right_hand_bars=[], left_hand_bars=[]),
        hand=Hand.RIGHT,
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
    )

    assert [token.degree for token in tokens if isinstance(token, NoteToken)] == [1, 3, 5]


def test_unified_stream_adds_join_suffixes_for_chord_notes(duration_vocabulary: DurationVocabulary) -> None:
    score = _score(
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
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
    )

    assert isinstance(tokenized_bars[0][0], HandToken)
    assert sum(isinstance(token, JoinWithPreviousToken) for token in tokenized_bars[0]) == 2


def test_unified_stream_rejects_overlapping_non_chord_notes(duration_vocabulary: DurationVocabulary) -> None:
    score = _score(
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
        _tokenize_unified_stream(
            score=score,
            scale_type=ScaleType.MAJOR,
            duration_vocabulary=duration_vocabulary,
        )
