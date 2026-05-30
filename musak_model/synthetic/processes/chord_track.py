from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, model_validator

from musak_model.harmony.schema import TRIAD_QUALITY_BY_INTERVALS, Chord
from musak_model.tokens.schema import SCALE_INTERVALS, ScaleType
from musak_shared.elements import HARMONIC_FUNCTION_BY_DEGREE, PITCHES_PER_OCTAVE, HarmonicFunction

_PROBABILITY_SUM_TOLERANCE: Final[float] = 1e-9

_DEFAULT_FUNCTIONAL_STRENGTH: Final[float] = 0.7
_DESTINATION_FLOOR: Final[float] = 0.1
_INITIAL_TONIC_BONUS: Final[float] = 1.0
_FUNCTION_PREFERENCE: Final[dict[HarmonicFunction, dict[HarmonicFunction, float]]] = {
    HarmonicFunction.TONIC: {
        HarmonicFunction.TONIC: 1.0,
        HarmonicFunction.PREDOMINANT: 3.0,
        HarmonicFunction.DOMINANT: 2.0,
    },
    HarmonicFunction.PREDOMINANT: {
        HarmonicFunction.TONIC: 1.0,
        HarmonicFunction.PREDOMINANT: 1.0,
        HarmonicFunction.DOMINANT: 4.0,
    },
    HarmonicFunction.DOMINANT: {
        HarmonicFunction.TONIC: 5.0,
        HarmonicFunction.PREDOMINANT: 1.0,
        HarmonicFunction.DOMINANT: 1.0,
    },
}


class ChordTransitionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_distribution: dict[Chord, float]
    transitions: dict[Chord, dict[Chord, float]]

    @model_validator(mode="after")
    def _validate_distributions(self) -> ChordTransitionModel:
        if not self.initial_distribution:
            raise ValueError("initial_distribution must be non-empty")

        _validate_distribution(self.initial_distribution, description="initial_distribution")
        if set(self.transitions.keys()) != set(self.initial_distribution.keys()):
            raise ValueError("transitions must be keyed by the same chords as initial_distribution")

        for source_chord, row in self.transitions.items():
            _validate_distribution(row, description=f"transitions[{source_chord!r}]")
            unknown = set(row.keys()) - set(self.initial_distribution.keys())
            if unknown:
                raise ValueError(
                    f"transitions[{source_chord!r}] references unknown chords: {sorted(unknown, key=repr)}"
                )

        return self


@dataclass(frozen=True)
class ChordTrackSampler:
    model: ChordTransitionModel

    def sample(
        self,
        *,
        length: int,
        rng: Generator,
    ) -> tuple[Chord, ...]:
        if length <= 0:
            raise ValueError("length must be positive")

        current = _categorical_choice(self.model.initial_distribution, rng=rng)
        track = [current]
        for _ in range(length - 1):
            current = _categorical_choice(self.model.transitions[current], rng=rng)
            track.append(current)

        return tuple(track)


def uniform_transition_model(
    chords: Sequence[Chord],
    *,
    self_transition_bias: float = 0.0,
) -> ChordTransitionModel:
    if not chords:
        raise ValueError("chords must be non-empty")

    if not 0.0 <= self_transition_bias <= 1.0:
        raise ValueError("self_transition_bias must be in [0, 1]")

    uniform_probability = 1.0 / len(chords)
    off_diagonal_probability = uniform_probability * (1.0 - self_transition_bias)
    diagonal_probability = off_diagonal_probability + self_transition_bias
    initial_distribution = {chord: uniform_probability for chord in chords}
    transitions = {
        source_chord: {
            destination_chord: diagonal_probability if destination_chord == source_chord else off_diagonal_probability
            for destination_chord in chords
        }
        for source_chord in chords
    }
    return ChordTransitionModel(
        initial_distribution=initial_distribution,
        transitions=transitions,
    )


def functional_transition_model(
    chords: Sequence[Chord],
    *,
    scale_type: ScaleType,
    strength: float = _DEFAULT_FUNCTIONAL_STRENGTH,
    self_transition_bias: float = 0.0,
) -> ChordTransitionModel:
    if not chords:
        raise ValueError("chords must be non-empty")

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    if not 0.0 <= self_transition_bias <= 1.0:
        raise ValueError("self_transition_bias must be in [0, 1]")

    function_by_chord = _function_by_chord(chords, scale_type=scale_type)
    uniform_probability = 1.0 / len(chords)
    initial_functional = _tonic_weighted_distribution(chords, function_by_chord=function_by_chord)
    initial_distribution = {
        chord: (1.0 - strength) * uniform_probability + strength * initial_functional[chord] for chord in chords
    }
    transitions = {
        source: _blended_transition_row(
            source,
            chords=chords,
            function_by_chord=function_by_chord,
            uniform_probability=uniform_probability,
            strength=strength,
            self_transition_bias=self_transition_bias,
        )
        for source in chords
    }
    return ChordTransitionModel(initial_distribution=initial_distribution, transitions=transitions)


def _function_by_chord(chords: Sequence[Chord], *, scale_type: ScaleType) -> dict[Chord, HarmonicFunction]:
    chord_set = set(chords)
    scale_size = len(SCALE_INTERVALS[scale_type])
    function_by_chord: dict[Chord, HarmonicFunction] = {}
    for degree, function in HARMONIC_FUNCTION_BY_DEGREE.items():
        if degree > scale_size:
            continue

        triad = _natural_triad(scale_type, degree)
        if triad in chord_set:
            function_by_chord[triad] = function

    return function_by_chord


def _natural_triad(scale_type: ScaleType, degree: int) -> Chord:
    intervals = SCALE_INTERVALS[scale_type]
    scale_size = len(intervals)
    root_index = degree - 1
    root_semitone = intervals[root_index]
    third = (intervals[(root_index + 2) % scale_size] - root_semitone) % PITCHES_PER_OCTAVE
    fifth = (intervals[(root_index + 4) % scale_size] - root_semitone) % PITCHES_PER_OCTAVE
    return Chord(root_degree=degree, root_accidental=0, quality=TRIAD_QUALITY_BY_INTERVALS[(third, fifth)])


def _tonic_weighted_distribution(
    chords: Sequence[Chord], *, function_by_chord: dict[Chord, HarmonicFunction]
) -> dict[Chord, float]:
    weights = {
        chord: _DESTINATION_FLOOR
        + (_INITIAL_TONIC_BONUS if function_by_chord.get(chord) == HarmonicFunction.TONIC else 0.0)
        for chord in chords
    }
    total = sum(weights.values())
    return {chord: weight / total for chord, weight in weights.items()}


def _blended_transition_row(
    source: Chord,
    *,
    chords: Sequence[Chord],
    function_by_chord: dict[Chord, HarmonicFunction],
    uniform_probability: float,
    strength: float,
    self_transition_bias: float,
) -> dict[Chord, float]:
    functional = _functional_destination_weights(source, chords=chords, function_by_chord=function_by_chord)
    row: dict[Chord, float] = {}
    for destination in chords:
        blended = (1.0 - strength) * uniform_probability + strength * functional[destination]
        row[destination] = (1.0 - self_transition_bias) * blended + (
            self_transition_bias if destination == source else 0.0
        )

    return row


def _functional_destination_weights(
    source: Chord,
    *,
    chords: Sequence[Chord],
    function_by_chord: dict[Chord, HarmonicFunction],
) -> dict[Chord, float]:
    source_function = function_by_chord.get(source)
    weights = {chord: _DESTINATION_FLOOR for chord in chords}
    if source_function is not None:
        for chord in chords:
            destination_function = function_by_chord.get(chord)
            if destination_function is not None:
                weights[chord] += _FUNCTION_PREFERENCE[source_function][destination_function]

    total = sum(weights.values())
    return {chord: weight / total for chord, weight in weights.items()}


def _categorical_choice(
    distribution: dict[Chord, float],
    *,
    rng: Generator,
) -> Chord:
    chords = tuple(distribution.keys())
    probabilities = np.fromiter(distribution.values(), dtype=np.float64, count=len(chords))
    return chords[int(rng.choice(len(chords), p=probabilities))]


def _validate_distribution(
    distribution: dict[Chord, float],
    *,
    description: str,
) -> None:
    if not distribution:
        raise ValueError(f"{description} must be non-empty")

    if any(probability < 0 for probability in distribution.values()):
        raise ValueError(f"{description} contains negative probabilities")

    total = sum(distribution.values())
    if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"{description} probabilities must sum to 1, got {total}")
