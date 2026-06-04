from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.evaluation.coherence import coherence_metrics
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, ScaleType, Token


def _segment(
    tokens: list[Token],
    *,
    bar_count: int,
    bar_durations: tuple[Fraction, ...] | None = None,
) -> Segment:
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=bar_count,
            bar_durations=bar_durations,
            window_start_bar=0,
            source_file=Path("test"),
            difficulty_level=None,
        ),
    )


def _note(degree: int, duration_id: int, *, octave_offset: int = 0) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=octave_offset, duration_id=duration_id)


def test_coherence_metrics_detect_static_bass_and_final_closure(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, quarter),
            _note(2, quarter),
            _note(3, quarter),
            _note(1, quarter),
            HandToken(hand=Hand.LEFT),
            _note(1, whole),
            BarToken(),
            EndToken(),
        ],
        bar_count=1,
    )

    metrics = coherence_metrics([segment], duration_vocabulary=duration_vocabulary)

    assert metrics["coherence/count/samples"] == 1.0
    assert metrics["coherence/count/note_events"] == 5.0
    assert metrics["coherence/count/long_note_events"] == 1.0
    assert metrics["coherence/count/whole_bar_note_events"] == 1.0
    assert metrics["coherence/count/whole_note_or_longer_events"] == 1.0
    assert metrics["coherence/rate/samples_with_long_note"] == 1.0
    assert metrics["coherence/rate/samples_with_whole_bar_note"] == 1.0
    assert metrics["coherence/rate/samples_with_whole_note_or_longer"] == 1.0
    assert metrics["coherence/rate/whole_bar_note_events"] == 0.2
    assert metrics["coherence/rate/whole_note_or_longer_events"] == 0.2
    assert metrics["coherence/rate/left_long_note_events"] == 1.0
    assert metrics["coherence/rate/left_whole_bar_note_events"] == 1.0
    assert metrics["coherence/rate/left_whole_note_or_longer_events"] == 1.0
    assert metrics["coherence/rate/right_long_note_events"] == 0.0
    assert metrics["coherence/rate/right_whole_bar_note_events"] == 0.0
    assert metrics["coherence/rate/right_whole_note_or_longer_events"] == 0.0
    assert metrics["coherence/rate/static_long_left_under_right_motion"] == 1.0
    assert metrics["coherence/rate/stepwise_motion"] == 2 / 3
    assert metrics["coherence/rate/direction_change"] == 0.5
    assert metrics["coherence/rate/answered_onsets"] == 0.2
    assert metrics["coherence/rate/synchronized_onsets"] == 0.25
    assert metrics["coherence/rate/final_activity"] == 1.0
    assert metrics["coherence/rate/final_both_hands_active"] == 1.0
    assert metrics["coherence/rate/final_left_root_support"] == 1.0
    assert metrics["coherence/rate/final_right_tonic_closure"] == 1.0
    assert metrics["coherence/rate/final_long_note"] == 1.0


def test_coherence_metrics_distinguish_whole_bar_from_whole_note_duration(
    duration_vocabulary: DurationVocabulary,
) -> None:
    half = duration_vocabulary.require_duration_id(Fraction(1, 2))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, half),
            BarToken(),
            EndToken(),
        ],
        bar_count=1,
        bar_durations=(Fraction(1, 2),),
    )

    metrics = coherence_metrics([segment], duration_vocabulary=duration_vocabulary)

    assert metrics["coherence/count/whole_bar_note_events"] == 1.0
    assert metrics["coherence/count/whole_note_or_longer_events"] == 0.0
    assert metrics["coherence/rate/samples_with_whole_bar_note"] == 1.0
    assert metrics["coherence/rate/samples_with_whole_note_or_longer"] == 0.0
    assert metrics["coherence/rate/right_whole_bar_note_events"] == 1.0
    assert metrics["coherence/rate/right_whole_note_or_longer_events"] == 0.0


def test_coherence_metrics_count_large_leap_recovery(duration_vocabulary: DurationVocabulary) -> None:
    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    segment = _segment(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, quarter),
            _note(1, quarter, octave_offset=1),
            _note(7, quarter),
            BarToken(),
            EndToken(),
        ],
        bar_count=1,
    )

    metrics = coherence_metrics([segment], duration_vocabulary=duration_vocabulary)

    assert metrics["coherence/count/melodic_intervals"] == 2.0
    assert metrics["coherence/count/large_leaps"] == 1.0
    assert metrics["coherence/rate/large_leap"] == 0.5
    assert metrics["coherence/rate/large_leap_recovery"] == 1.0
