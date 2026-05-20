from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.data.augmentation import double_durations, shift_register
from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, ScaleType


def _metadata() -> SegmentMetadata:
    return SegmentMetadata(
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=1,
        window_start_bar=0,
        source_file=Path("score.mxl"),
    )


def test_shift_register_updates_canonical_unified_tokens(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=1, duration_id=quarter_id),
            EndToken(),
        ],
        metadata=_metadata(),
    )

    shifted = shift_register(segment, offset=-1)

    note_tokens = [token for token in shifted.tokens if isinstance(token, NoteToken)]
    assert [token.octave_offset for token in note_tokens] == [-1, 0]


def test_double_durations_rejects_bars_that_exceed_measure_duration(
    duration_vocabulary: DurationVocabulary,
) -> None:
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=half_id),
            NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=half_id),
            BarToken(),
            EndToken(),
        ],
        metadata=_metadata(),
    )

    with pytest.raises(ValueError, match="exceeds measure duration"):
        double_durations(segment, duration_vocabulary=duration_vocabulary)
