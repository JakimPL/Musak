from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Final

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.artifacts import FIGURE_ALL_DIR_NAME, FIGURE_COUNTS_NAME
from musak_model.n_grams.profile.io import read_figure_counts_csv
from musak_model.paths import DEFAULT_TRAINING_FIGURE_DIR
from musak_model.tokens.schema import Hand, ScaleType

_DEFAULT_COMMONNESS_BIAS: Final[float] = 1.0


@dataclass(frozen=True)
class FigureVocabularyGroup:
    scale_type: ScaleType
    hand: Hand
    n: int


@dataclass(frozen=True)
class FigureVocabularyEntry:
    group: FigureVocabularyGroup
    figure: FigureNGram
    count: int


@dataclass(frozen=True)
class FigureVocabulary:
    entries: tuple[FigureVocabularyEntry, ...]

    @property
    def unique_count(self) -> int:
        return len(self.entries)

    @property
    def total_count(self) -> int:
        return sum(entry.count for entry in self.entries)

    @classmethod
    def from_counts(cls, counts: FigureNGramCountsByScale) -> FigureVocabulary:
        entries: list[FigureVocabularyEntry] = []
        for scale_type, counts_by_hand in sorted(counts.items(), key=lambda item: item[0].value):
            for hand, counts_by_n in sorted(counts_by_hand.items(), key=lambda item: item[0].value):
                for n, figure_counts in sorted(counts_by_n.items()):
                    for figure, count in figure_counts.most_common():
                        entries.append(
                            FigureVocabularyEntry(
                                group=FigureVocabularyGroup(scale_type=scale_type, hand=hand, n=n),
                                figure=figure,
                                count=count,
                            )
                        )

        return cls(entries=tuple(entries))

    def filter(
        self,
        *,
        scale_type: ScaleType | None = None,
        hand: Hand | None = None,
        n: int | None = None,
        monophonic: bool | None = None,
        chords_only: bool | None = None,
        in_scale: bool | None = None,
        min_count: int = 1,
    ) -> FigureVocabulary:
        if min_count <= 0:
            raise ValueError("min_count must be positive")

        return FigureVocabulary(
            entries=tuple(
                entry
                for entry in self.entries
                if _entry_matches(
                    entry,
                    scale_type=scale_type,
                    hand=hand,
                    n=n,
                    monophonic=monophonic,
                    chords_only=chords_only,
                    in_scale=in_scale,
                    min_count=min_count,
                )
            )
        )

    def groups(self) -> tuple[FigureVocabularyGroup, ...]:
        return tuple(dict.fromkeys(entry.group for entry in self.entries))

    def length_distribution(self) -> dict[int, float]:
        totals: dict[int, int] = {}
        for entry in self.entries:
            totals[entry.group.n] = totals.get(entry.group.n, 0) + entry.count

        total_count = sum(totals.values())
        if total_count == 0:
            return {}

        return {n: count / total_count for n, count in sorted(totals.items())}

    def sample(
        self,
        *,
        rng: Random,
        commonness_bias: float = _DEFAULT_COMMONNESS_BIAS,
    ) -> FigureVocabularyEntry:
        if commonness_bias < 0:
            raise ValueError("commonness_bias must be non-negative")

        if not self.entries:
            raise ValueError("cannot sample from an empty figure vocabulary")

        weights = tuple(_entry_weight(entry, commonness_bias=commonness_bias) for entry in self.entries)
        return _weighted_choice(self.entries, weights=weights, rng=rng)


def load_figure_vocabulary(path: Path) -> FigureVocabulary:
    return FigureVocabulary.from_counts(read_figure_counts_csv(resolve_figure_counts_path(path)))


def load_figure_split_vocabulary(
    *,
    split_key: str,
    split_name: str,
    artifact_root: Path = DEFAULT_TRAINING_FIGURE_DIR,
) -> FigureVocabulary:
    return load_figure_vocabulary(artifact_root / split_key / split_name)


def resolve_figure_counts_path(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = (
        path / FIGURE_COUNTS_NAME,
        path / FIGURE_ALL_DIR_NAME / FIGURE_COUNTS_NAME,
        path / "figure" / FIGURE_ALL_DIR_NAME / FIGURE_COUNTS_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    candidate_text = ", ".join(candidate.as_posix() for candidate in candidates)
    raise FileNotFoundError(f"could not find figure counts CSV at {path} or one of: {candidate_text}")


def _entry_matches(
    entry: FigureVocabularyEntry,
    *,
    scale_type: ScaleType | None,
    hand: Hand | None,
    n: int | None,
    monophonic: bool | None,
    chords_only: bool | None,
    in_scale: bool | None,
    min_count: int,
) -> bool:
    return (
        entry.count >= min_count
        and (scale_type is None or entry.group.scale_type == scale_type)
        and (hand is None or entry.group.hand == hand)
        and (n is None or entry.group.n == n)
        and (monophonic is None or entry.figure.monophonic == monophonic)
        and (chords_only is None or entry.figure.chords_only == chords_only)
        and (in_scale is None or entry.figure.in_scale == in_scale)
    )


def _entry_weight(entry: FigureVocabularyEntry, *, commonness_bias: float) -> float:
    if commonness_bias == 0:
        return 1.0

    return float(pow(float(entry.count), commonness_bias))


def _weighted_choice(
    entries: tuple[FigureVocabularyEntry, ...],
    *,
    weights: Iterable[float],
    rng: Random,
) -> FigureVocabularyEntry:
    cumulative_weights: list[float] = []
    total_weight = 0.0
    for weight in weights:
        total_weight += weight
        cumulative_weights.append(total_weight)

    if total_weight <= 0:
        raise ValueError("cannot sample from figure vocabulary with no positive weights")

    threshold = rng.random() * total_weight
    for index, cumulative_weight in enumerate(cumulative_weights):
        if threshold < cumulative_weight:
            return entries[index]

    return entries[-1]
