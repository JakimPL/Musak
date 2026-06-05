from collections.abc import Sequence
from fractions import Fraction

from musak_model.data.schema import Segment
from musak_model.decoder import PianoRollEvent, segment_to_piano_roll_events
from musak_model.rhythm_refiner.schema import (
    CoactivityState,
    RhythmCellState,
    RhythmGridCell,
    RhythmGridConfig,
    RhythmGridFrame,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand


def rhythm_grid_from_segment(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    config: RhythmGridConfig,
) -> RhythmGridFrame:
    bar_durations = _segment_bar_durations(segment)
    cells = _rhythm_grid_cells(bar_durations, grid_denominator=config.grid_denominator)
    events = tuple(segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary))
    right_hand_states = _hand_cell_states(cells, events, hand=Hand.RIGHT)
    left_hand_states = _hand_cell_states(cells, events, hand=Hand.LEFT)
    return RhythmGridFrame(
        config=config,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
        bar_durations=bar_durations,
        cells=cells,
        right_hand_states=right_hand_states,
        left_hand_states=left_hand_states,
        coactivity_states=tuple(
            _coactivity_state(right_state, left_state)
            for right_state, left_state in zip(right_hand_states, left_hand_states, strict=True)
        ),
    )


def _segment_bar_durations(segment: Segment) -> tuple[Fraction, ...]:
    if segment.metadata.bar_durations is not None:
        if len(segment.metadata.bar_durations) < segment.bar_count:
            raise ValueError("segment bar_durations must cover every segment bar")
        return segment.metadata.bar_durations[: segment.bar_count]

    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    return tuple(measure_duration for _ in range(segment.bar_count))


def _rhythm_grid_cells(
    bar_durations: Sequence[Fraction],
    *,
    grid_denominator: int,
) -> tuple[RhythmGridCell, ...]:
    cell_duration = Fraction(1, grid_denominator)
    cell_counts = tuple(_bar_cell_count(bar_duration, cell_duration=cell_duration) for bar_duration in bar_durations)
    total_cell_count = sum(cell_counts)
    cells: list[RhythmGridCell] = []
    bar_start = Fraction(0)

    for bar_index, (bar_duration, bar_cell_count) in enumerate(zip(bar_durations, cell_counts, strict=True)):
        for cell_index in range(bar_cell_count):
            bar_relative_start = cell_index * cell_duration
            bar_relative_end = bar_relative_start + cell_duration
            global_cell_index = len(cells)
            cells.append(
                RhythmGridCell(
                    global_cell_index=global_cell_index,
                    bar_index=bar_index,
                    cell_index=cell_index,
                    start=bar_start + bar_relative_start,
                    end=bar_start + bar_relative_end,
                    bar_relative_start=bar_relative_start,
                    bar_relative_end=bar_relative_end,
                    metrical_offset=bar_relative_start,
                    distance_to_end=total_cell_count - global_cell_index - 1,
                )
            )
        bar_start += bar_duration

    return tuple(cells)


def _bar_cell_count(bar_duration: Fraction, *, cell_duration: Fraction) -> int:
    cell_count = bar_duration / cell_duration
    if cell_count.denominator != 1:
        raise ValueError(f"bar duration {bar_duration} cannot be represented on rhythm grid {cell_duration}")
    return cell_count.numerator


def _hand_cell_states(
    cells: Sequence[RhythmGridCell],
    events: Sequence[PianoRollEvent],
    *,
    hand: Hand,
) -> tuple[RhythmCellState, ...]:
    hand_events = tuple(event for event in events if event.hand == hand)
    return tuple(_cell_state(cell, hand_events) for cell in cells)


def _cell_state(cell: RhythmGridCell, events: Sequence[PianoRollEvent]) -> RhythmCellState:
    if any(cell.start <= event.start < cell.end for event in events):
        return RhythmCellState.ONSET
    if any(event.start < cell.start and event.end > cell.start for event in events):
        return RhythmCellState.SUSTAIN
    return RhythmCellState.REST


def _coactivity_state(right_state: RhythmCellState, left_state: RhythmCellState) -> CoactivityState:
    if right_state == RhythmCellState.REST and left_state == RhythmCellState.REST:
        return CoactivityState.SILENT
    if right_state == RhythmCellState.ONSET and left_state == RhythmCellState.ONSET:
        return CoactivityState.BOTH_SYNCHRONIZED
    if right_state == RhythmCellState.ONSET and left_state == RhythmCellState.SUSTAIN:
        return CoactivityState.RIGHT_ONSET_LEFT_SUSTAIN
    if right_state == RhythmCellState.SUSTAIN and left_state == RhythmCellState.ONSET:
        return CoactivityState.LEFT_ONSET_RIGHT_SUSTAIN
    if right_state == RhythmCellState.SUSTAIN and left_state == RhythmCellState.SUSTAIN:
        return CoactivityState.BOTH_SUSTAIN
    if right_state != RhythmCellState.REST and left_state == RhythmCellState.REST:
        return CoactivityState.RIGHT_ONLY
    if left_state != RhythmCellState.REST and right_state == RhythmCellState.REST:
        return CoactivityState.LEFT_ONLY
    return CoactivityState.BOTH_ACTIVE
