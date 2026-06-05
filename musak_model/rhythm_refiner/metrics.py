from collections import Counter
from collections.abc import Callable, Iterable

from musak_model.rhythm_refiner.schema import CoactivityState, RhythmCellState, RhythmGridFrame
from musak_model.tokens.schema import Hand


def rhythm_grid_metric_values(
    frame: RhythmGridFrame,
    *,
    metric_prefix: str = "rhythm_refiner/grid",
) -> dict[str, float]:
    cell_count = len(frame.cells)
    metrics = {f"{metric_prefix}/count/cells": float(cell_count)}
    metrics.update(
        _state_rate_metrics(
            frame.right_hand_states,
            metric_prefix=f"{metric_prefix}/right_hand",
        )
    )
    metrics.update(
        _state_rate_metrics(
            frame.left_hand_states,
            metric_prefix=f"{metric_prefix}/left_hand",
        )
    )
    metrics.update(_coactivity_rate_metrics(frame.coactivity_states, metric_prefix=f"{metric_prefix}/coactivity"))
    metrics[f"{metric_prefix}/rate/one_hand_active"] = _one_hand_active_rate(frame)
    metrics[f"{metric_prefix}/rate/both_hands_active"] = _both_hands_active_rate(frame)
    metrics[f"{metric_prefix}/rate/synchronized_onset"] = _state_fraction(
        frame.coactivity_states,
        CoactivityState.BOTH_SYNCHRONIZED,
    )
    return metrics


def _state_rate_metrics(states: Iterable[RhythmCellState], *, metric_prefix: str) -> dict[str, float]:
    state_tuple = tuple(states)
    return {
        f"{metric_prefix}/rate/{state.value}": _state_fraction(state_tuple, state)
        for state in (RhythmCellState.REST, RhythmCellState.ONSET, RhythmCellState.SUSTAIN)
    }


def _coactivity_rate_metrics(states: Iterable[CoactivityState], *, metric_prefix: str) -> dict[str, float]:
    state_tuple = tuple(states)
    return {f"{metric_prefix}/rate/{state.value}": _state_fraction(state_tuple, state) for state in CoactivityState}


def _one_hand_active_rate(frame: RhythmGridFrame) -> float:
    return _matching_cell_fraction(
        right_states=frame.states_for_hand(Hand.RIGHT),
        left_states=frame.states_for_hand(Hand.LEFT),
        predicate=lambda right_state, left_state: (right_state == RhythmCellState.REST)
        != (left_state == RhythmCellState.REST),
    )


def _both_hands_active_rate(frame: RhythmGridFrame) -> float:
    return _matching_cell_fraction(
        right_states=frame.states_for_hand(Hand.RIGHT),
        left_states=frame.states_for_hand(Hand.LEFT),
        predicate=lambda right_state, left_state: RhythmCellState.REST not in (right_state, left_state),
    )


def _matching_cell_fraction(
    *,
    right_states: tuple[RhythmCellState, ...],
    left_states: tuple[RhythmCellState, ...],
    predicate: Callable[[RhythmCellState, RhythmCellState], bool],
) -> float:
    if not right_states:
        return 0.0

    matching_count = sum(
        bool(predicate(right_state, left_state))
        for right_state, left_state in zip(right_states, left_states, strict=True)
    )
    return matching_count / len(right_states)


def _state_fraction[T](states: tuple[T, ...], state: T) -> float:
    if not states:
        return 0.0

    return Counter(states)[state] / len(states)
