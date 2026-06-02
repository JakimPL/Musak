from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.random import Generator

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.artifacts import FIGURE_ALL_DIR_NAME, FIGURE_BASE_DURATIONS_NAME
from musak_model.n_grams.profile.io import read_base_duration_counts
from musak_model.paths import DEFAULT_TRAINING_FIGURE_DIRECTORY
from musak_model.tokens.duration import DurationVocabulary
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


def fitting_base_durations(
    figure: FigureNGram,
    candidates: Sequence[BaseDurationWeight],
    *,
    remaining: Fraction,
    duration_vocabulary: DurationVocabulary,
) -> list[BaseDurationWeight]:
    normalized_durations = [duration for _, duration in figure.onsets]
    span = sum(normalized_durations, Fraction(0))
    fitting: list[BaseDurationWeight] = []
    for base_duration, count in candidates:
        if span * base_duration > remaining:
            continue

        if all(
            duration_vocabulary.duration_id_or_none(normalized * base_duration) is not None
            for normalized in normalized_durations
        ):
            fitting.append((base_duration, count))

    return fitting


def choose_base_duration(
    figure: FigureNGram,
    candidates: Sequence[BaseDurationWeight],
    *,
    density_offset: float,
    remaining: Fraction,
    duration_vocabulary: DurationVocabulary,
) -> Fraction | None:
    fitting = fitting_base_durations(figure, candidates, remaining=remaining, duration_vocabulary=duration_vocabulary)
    if not fitting:
        return None

    target_log = _weighted_median_log(fitting) + density_offset
    return min(fitting, key=lambda item: abs(math.log2(float(item[0])) - target_log))[0]


def _weighted_median_log(candidates: Sequence[BaseDurationWeight]) -> float:
    ordered = sorted(candidates, key=lambda item: item[0])
    total = sum(count for _, count in ordered)
    cumulative = 0
    for base_duration, count in ordered:
        cumulative += count
        if cumulative * 2 >= total:
            return math.log2(float(base_duration))

    return math.log2(float(ordered[-1][0]))


def load_base_duration_distribution(path: Path) -> BaseDurationDistribution:
    counts_by_group = read_base_duration_counts(resolve_base_durations_path(path))
    weights_by_group = {
        group: tuple(sorted(duration_counts.items())) for group, duration_counts in counts_by_group.items()
    }
    return BaseDurationDistribution(weights_by_group=weights_by_group)


def load_base_duration_split_distribution(
    *,
    split_key: str,
    split_name: str,
    artifact_root: Path = DEFAULT_TRAINING_FIGURE_DIRECTORY,
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
    raise FileNotFoundError(f"could not find base durations table at {path} or one of: {candidate_text}")
