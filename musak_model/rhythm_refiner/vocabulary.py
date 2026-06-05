from typing import Final

from musak_model.rhythm_refiner.schema import CoactivityState, RhythmCellState

RHYTHM_INPUT_UNKNOWN_ID: Final[int] = 0
RHYTHM_INPUT_STATE_COUNT: Final[int] = 4
RHYTHM_TARGET_STATE_COUNT: Final[int] = 3
RHYTHM_TARGET_IGNORE_ID: Final[int] = -100
COACTIVITY_TARGET_IGNORE_ID: Final[int] = -100

_RHYTHM_INPUT_STATE_IDS: Final[dict[RhythmCellState, int]] = {
    RhythmCellState.UNKNOWN: RHYTHM_INPUT_UNKNOWN_ID,
    RhythmCellState.REST: 1,
    RhythmCellState.ONSET: 2,
    RhythmCellState.SUSTAIN: 3,
}
_RHYTHM_TARGET_STATE_IDS: Final[dict[RhythmCellState, int]] = {
    RhythmCellState.REST: 0,
    RhythmCellState.ONSET: 1,
    RhythmCellState.SUSTAIN: 2,
}
COACTIVITY_STATES: Final[tuple[CoactivityState, ...]] = tuple(CoactivityState)
COACTIVITY_TARGET_STATE_COUNT: Final[int] = len(COACTIVITY_STATES)
_COACTIVITY_STATE_IDS: Final[dict[CoactivityState, int]] = {
    state: index for index, state in enumerate(COACTIVITY_STATES)
}


def rhythm_input_state_id(state: RhythmCellState) -> int:
    return _RHYTHM_INPUT_STATE_IDS[state]


def rhythm_target_state_id(state: RhythmCellState) -> int:
    if state == RhythmCellState.UNKNOWN:
        raise ValueError("unknown rhythm cell state is not a valid training target")

    return _RHYTHM_TARGET_STATE_IDS[state]


def coactivity_target_state_id(state: CoactivityState) -> int:
    return _COACTIVITY_STATE_IDS[state]
