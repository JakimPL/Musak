from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.random import Generator

from musak_model.n_grams.profile.artifacts import FIGURE_ALL_DIR_NAME, FIGURE_BASE_DURATIONS_NAME
from musak_model.n_grams.profile.io import read_base_duration_counts_csv
from musak_model.paths import DEFAULT_TRAINING_FIGURE_DIR
from musak_model.tokens.schema import Hand, ScaleType

type BaseDurationGroup = tuple[ScaleType, Hand, int]
type BaseDurationWeight = tuple[Fraction, int]


@dataclass(frozen=True)
class BaseDurationDistribution:
    weights_by_group: Mapping[BaseDurationGroup, tuple[BaseDurationWeight, ...]]

    def candidates(
        self,
        *,
        scale_type: ScaleType,
        hand: Hand,
        figure_length: int,
    ) -> tuple[BaseDurationWeight, ...]:
        return self.weights_by_group.get((scale_type, hand, figure_length), ())

    def sample(
        self,
        *,
        scale_type: ScaleType,
        hand: Hand,
        figure_length: int,
        rng: Generator,
    ) -> Fraction:
        candidates = self.candidates(scale_type=scale_type, hand=hand, figure_length=figure_length)
        if not candidates:
            raise ValueError(f"no base durations for ({scale_type.value}, {hand.value}, {figure_length})")

        return weighted_base_duration_choice(candidates, rng=rng)


def weighted_base_duration_choice(
    candidates: Sequence[BaseDurationWeight],
    *,
    rng: Generator,
) -> Fraction:
    if not candidates:
        raise ValueError("candidates must be non-empty")

    durations = [duration for duration, _ in candidates]
    weights = np.fromiter((count for _, count in candidates), dtype=np.float64, count=len(candidates))
    probabilities = weights / weights.sum()
    return durations[int(rng.choice(len(durations), p=probabilities))]


def load_base_duration_distribution(path: Path) -> BaseDurationDistribution:
    counts_by_group = read_base_duration_counts_csv(resolve_base_durations_path(path))
    weights_by_group = {
        group: tuple(sorted(duration_counts.items())) for group, duration_counts in counts_by_group.items()
    }
    return BaseDurationDistribution(weights_by_group=weights_by_group)


def load_base_duration_split_distribution(
    *,
    split_key: str,
    split_name: str,
    artifact_root: Path = DEFAULT_TRAINING_FIGURE_DIR,
) -> BaseDurationDistribution:
    return load_base_duration_distribution(artifact_root / split_key / split_name)


def resolve_base_durations_path(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = (
        path / FIGURE_BASE_DURATIONS_NAME,
        path / FIGURE_ALL_DIR_NAME / FIGURE_BASE_DURATIONS_NAME,
        path / "figure" / FIGURE_ALL_DIR_NAME / FIGURE_BASE_DURATIONS_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    candidate_text = ", ".join(candidate.as_posix() for candidate in candidates)
    raise FileNotFoundError(f"could not find base durations CSV at {path} or one of: {candidate_text}")
