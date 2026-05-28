from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, model_validator

from musak_model.synthetic.harmony.schema import Chord

_PROBABILITY_SUM_TOLERANCE: Final[float] = 1e-9


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
