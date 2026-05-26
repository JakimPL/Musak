from fractions import Fraction

import pytest

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote, ParsedScore
from musak_model.data.segmenter.streams import tokenize_unified_stream
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import HandToken, JoinWithPreviousToken, NoteToken, ScaleType


def _score(*, right_hand_bars: list[ParsedBar], left_hand_bars: list[ParsedBar]) -> ParsedScore:
    return ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=right_hand_bars,
        left_hand_bars=left_hand_bars,
    )


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

    tokenized_bars = tokenize_unified_stream(score=score, duration_vocabulary=duration_vocabulary)

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
        tokenize_unified_stream(score=score, duration_vocabulary=duration_vocabulary)


def test_unified_stream_rejects_overlap_smaller_than_shortest_supported_duration(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = _score(
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
        tokenize_unified_stream(score=score, duration_vocabulary=duration_vocabulary)


def test_cleaned_unified_stream_preserves_notes_after_truncating_overlaps(
    duration_vocabulary: DurationVocabulary,
) -> None:
    score = _score(
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

    tokenized_bars = tokenize_unified_stream(
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
