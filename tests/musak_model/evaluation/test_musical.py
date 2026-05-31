from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.evaluation.musical import musical_metrics
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
)


def _segment(tokens: list[Token], *, bar_count: int) -> Segment:
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=bar_count,
            window_start_bar=0,
            source_file=Path("test"),
            difficulty_level=None,
        ),
    )


def _note(degree: int, duration_id: int) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=duration_id)


def test_harmonic_consonance_rate_counts_coincident_onset_intervals(duration_vocabulary: DurationVocabulary) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, whole),
            HandToken(hand=Hand.LEFT),
            _note(1, whole),  # octave below the right hand -> consonant
            BarToken(),
            HandToken(hand=Hand.RIGHT),
            _note(1, whole),
            HandToken(hand=Hand.LEFT),
            _note(2, whole),  # interval class 10 (minor seventh) -> dissonant
            BarToken(),
            EndToken(),
        ],
        bar_count=2,
    )

    metrics = musical_metrics([segment], duration_vocabulary=duration_vocabulary)

    assert metrics["musical/count/coincident_onset_pairs"] == 2.0
    assert metrics["musical/rate/harmonic_consonance"] == 0.5


def test_register_autocorrelation_high_for_ascending_line(duration_vocabulary: DurationVocabulary) -> None:
    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, quarter),
            _note(2, quarter),
            _note(3, quarter),
            _note(4, quarter),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=whole),
            BarToken(),
            EndToken(),
        ],
        bar_count=1,
    )

    metrics = musical_metrics([segment], duration_vocabulary=duration_vocabulary)

    assert metrics["musical/mean/register_lag1_autocorrelation"] > 0.9
    assert "musical/rate/harmonic_consonance" not in metrics


def test_no_coincident_onsets_and_too_few_notes_report_only_the_count(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, whole),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=whole),
            BarToken(),
            EndToken(),
        ],
        bar_count=1,
    )

    metrics = musical_metrics([segment], duration_vocabulary=duration_vocabulary)

    assert metrics["musical/count/coincident_onset_pairs"] == 0.0
    assert "musical/rate/harmonic_consonance" not in metrics
    assert "musical/mean/register_lag1_autocorrelation" not in metrics
