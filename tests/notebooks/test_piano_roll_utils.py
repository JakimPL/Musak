from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import ParsedBar, ParsedNote, ParsedScore, Segment, SegmentMetadata
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from notebooks.utils.piano_roll import (
    PitchSpelling,
    midi_pitch_name,
    parsed_score_piano_roll_dataframe,
    piano_roll_dataframe,
)


def _duration_vocabulary() -> DurationVocabulary:
    return DurationVocabulary(TokenizationConfig(shortest_duration=16, max_tuplets=(3,), max_dots=1))


def test_midi_pitch_name_uses_scientific_pitch_octaves() -> None:
    assert midi_pitch_name(60) == "C-4"
    assert midi_pitch_name(61) == "C#4"
    assert midi_pitch_name(61, pitch_spelling=PitchSpelling.FLATS) == "Db4"
    assert midi_pitch_name(58, pitch_spelling=PitchSpelling.FLATS) == "Bb3"


def test_segment_piano_roll_dataframe_includes_axis_and_token_fields() -> None:
    duration_vocabulary = _duration_vocabulary()
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=1, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=2,
            source_file=Path("score.mxl"),
        ),
    )

    row = piano_roll_dataframe(
        segment,
        duration_vocabulary=duration_vocabulary,
        pitch_spelling=PitchSpelling.SHARPS,
        bpm=120,
    ).iloc[0]

    assert row["pitch"] == "C#5"
    assert row["bar_start"] == 3.0
    assert row["bar_end"] == 3.25
    assert row["start_seconds"] == 0.0
    assert row["duration_fraction"] == "1:4"
    assert row["duration_seconds"] == 0.5
    assert row["token_index"] == 1
    assert row["token"] == "1♯(1:4)"


def test_parsed_score_piano_roll_dataframe_uses_pitch_spelling_without_token_fields() -> None:
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        mode="major",
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=61, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))],
            )
        ],
        left_hand_bars=[],
    )

    row = parsed_score_piano_roll_dataframe(score, pitch_spelling=PitchSpelling.FLATS, bpm=60).iloc[0]

    assert row["pitch"] == "Db4"
    assert row["bar_start"] == 1.25
    assert row["bar_end"] == 1.5
    assert row["start_seconds"] == 1.0
    assert row["duration_fraction"] == "1:4"
    assert row["duration_seconds"] == 1.0
    assert row["token"] is None
