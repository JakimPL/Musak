from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.rhythm_refiner import (
    CoactivityState,
    RhythmCellState,
    RhythmGridConfig,
    rhythm_grid_from_segment,
    rhythm_grid_metric_values,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    Hand,
    HandToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
)


def test_rhythm_grid_marks_onset_sustain_and_rest(duration_vocabulary: DurationVocabulary) -> None:
    whole_id = duration_vocabulary.require_duration_id(Fraction(1))
    frame = rhythm_grid_from_segment(
        _segment(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=whole_id),
                HandToken(hand=Hand.LEFT),
                RestToken(duration_id=whole_id),
                BarToken(),
            ]
        ),
        duration_vocabulary=duration_vocabulary,
        config=RhythmGridConfig(grid_denominator=4),
    )

    assert frame.right_hand_states == (
        RhythmCellState.ONSET,
        RhythmCellState.SUSTAIN,
        RhythmCellState.SUSTAIN,
        RhythmCellState.SUSTAIN,
    )
    assert frame.left_hand_states == (RhythmCellState.REST,) * 4
    assert frame.coactivity_states == (CoactivityState.RIGHT_ONLY,) * 4


def test_rhythm_grid_treats_joined_chord_as_single_activity_onset(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    frame = rhythm_grid_from_segment(
        _segment(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=quarter_id),
                _note(3, duration_id=quarter_id),
                JoinWithPreviousToken(),
            ]
        ),
        duration_vocabulary=duration_vocabulary,
        config=RhythmGridConfig(grid_denominator=4),
    )

    assert frame.right_hand_states == (
        RhythmCellState.ONSET,
        RhythmCellState.REST,
        RhythmCellState.REST,
        RhythmCellState.REST,
    )


def test_rhythm_grid_derives_hand_coactivity(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    half_id = duration_vocabulary.require_duration_id(Fraction(1, 2))
    whole_id = duration_vocabulary.require_duration_id(Fraction(1))
    frame = rhythm_grid_from_segment(
        _segment(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=whole_id),
                HandToken(hand=Hand.LEFT),
                RestToken(duration_id=quarter_id),
                _note(5, duration_id=quarter_id),
                RestToken(duration_id=half_id),
            ]
        ),
        duration_vocabulary=duration_vocabulary,
        config=RhythmGridConfig(grid_denominator=4),
    )

    assert frame.coactivity_states == (
        CoactivityState.RIGHT_ONLY,
        CoactivityState.LEFT_ONSET_RIGHT_SUSTAIN,
        CoactivityState.RIGHT_ONLY,
        CoactivityState.RIGHT_ONLY,
    )


def test_rhythm_grid_marks_synchronized_hand_onsets(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    dotted_half_id = duration_vocabulary.require_duration_id(Fraction(3, 4))
    frame = rhythm_grid_from_segment(
        _segment(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=quarter_id),
                RestToken(duration_id=dotted_half_id),
                HandToken(hand=Hand.LEFT),
                _note(5, duration_id=quarter_id),
                RestToken(duration_id=dotted_half_id),
            ]
        ),
        duration_vocabulary=duration_vocabulary,
        config=RhythmGridConfig(grid_denominator=4),
    )

    assert frame.coactivity_states == (
        CoactivityState.BOTH_SYNCHRONIZED,
        CoactivityState.SILENT,
        CoactivityState.SILENT,
        CoactivityState.SILENT,
    )


def test_rhythm_grid_uses_short_bar_duration(duration_vocabulary: DurationVocabulary) -> None:
    half_id = duration_vocabulary.require_duration_id(Fraction(1, 2))
    frame = rhythm_grid_from_segment(
        _segment(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=half_id),
                BarToken(),
            ],
            bar_durations=(Fraction(1, 2),),
        ),
        duration_vocabulary=duration_vocabulary,
        config=RhythmGridConfig(grid_denominator=4),
    )

    assert len(frame.cells) == 2
    assert frame.cells[-1].end == Fraction(1, 2)
    assert frame.right_hand_states == (RhythmCellState.ONSET, RhythmCellState.SUSTAIN)


def test_rhythm_grid_metric_values_report_state_and_coactivity_rates(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    dotted_half_id = duration_vocabulary.require_duration_id(Fraction(3, 4))
    frame = rhythm_grid_from_segment(
        _segment(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=quarter_id),
                RestToken(duration_id=dotted_half_id),
                HandToken(hand=Hand.LEFT),
                _note(5, duration_id=quarter_id),
                RestToken(duration_id=dotted_half_id),
            ]
        ),
        duration_vocabulary=duration_vocabulary,
        config=RhythmGridConfig(grid_denominator=4),
    )

    metrics = rhythm_grid_metric_values(frame, metric_prefix="test/rhythm")

    assert metrics["test/rhythm/count/cells"] == 4.0
    assert metrics["test/rhythm/right_hand/rate/onset"] == 0.25
    assert metrics["test/rhythm/left_hand/rate/onset"] == 0.25
    assert metrics["test/rhythm/coactivity/rate/both_synchronized"] == 0.25
    assert metrics["test/rhythm/rate/both_hands_active"] == 0.25


def _segment(
    tokens: list[Token],
    *,
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
            bar_count=1,
            bar_durations=bar_durations,
            window_start_bar=0,
            source_file=Path("sample.mxl"),
            difficulty_level=None,
        ),
    )


def _note(degree: int, *, duration_id: int) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=duration_id)
