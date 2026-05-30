from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.artifacts import FIGURE_ALL_DIR_NAME, FIGURE_COUNTS_NAME
from musak_model.n_grams.profile.io import AnchoredFigureCountsByGroup, read_anchor_figure_counts, read_figure_counts
from musak_model.paths import DEFAULT_TRAINING_FIGURE_DIR
from musak_model.tokens.schema import Hand, ScaleType


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


@dataclass(frozen=True)
class AnchoredFigureVocabularyEntry:
    group: FigureVocabularyGroup
    anchor_degree: int
    figure: FigureNGram
    count: int


@dataclass(frozen=True)
class AnchoredFigureVocabulary:
    entries: tuple[AnchoredFigureVocabularyEntry, ...]

    @classmethod
    def from_counts(cls, counts: AnchoredFigureCountsByGroup) -> AnchoredFigureVocabulary:
        entries: list[AnchoredFigureVocabularyEntry] = []
        for (scale_type, hand, n, anchor_degree, anchor_accidental), figure_counts in sorted(
            counts.items(), key=lambda item: (item[0][0].value, item[0][1].value, item[0][2:])
        ):
            if anchor_accidental != 0:
                continue

            for figure, count in figure_counts.most_common():
                entries.append(
                    AnchoredFigureVocabularyEntry(
                        group=FigureVocabularyGroup(scale_type=scale_type, hand=hand, n=n),
                        anchor_degree=anchor_degree,
                        figure=figure,
                        count=count,
                    )
                )

        return cls(entries=tuple(entries))


def load_figure_vocabulary(path: Path) -> FigureVocabulary:
    return FigureVocabulary.from_counts(read_figure_counts(resolve_figure_counts_path(path)))


def load_anchored_figure_vocabulary(path: Path) -> AnchoredFigureVocabulary:
    return AnchoredFigureVocabulary.from_counts(read_anchor_figure_counts(resolve_figure_counts_path(path)))


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
    raise FileNotFoundError(f"could not find figure counts table at {path} or one of: {candidate_text}")


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
