from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Final

import polars as pl

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.schema import FigureProfile, FigureSampleCounts
from musak_model.processing.io import JSON_INDENT
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.ratios import parse_ratio
from musak_shared.tables import read_table, write_table

SCALE_TYPE_COLUMN: Final[str] = "scale_type"
HAND_COLUMN: Final[str] = "hand"
N_COLUMN: Final[str] = "n"
COUNT_COLUMN: Final[str] = "count"
FIGURE_COLUMN: Final[str] = "figure"
BASE_DURATION_COLUMN: Final[str] = "base_duration"

FIGURE_COUNT_SCHEMA: Final[dict[str, pl.DataType]] = {
    SCALE_TYPE_COLUMN: pl.String(),
    HAND_COLUMN: pl.String(),
    N_COLUMN: pl.Int64(),
    COUNT_COLUMN: pl.Int64(),
    FIGURE_COLUMN: pl.String(),
}
BASE_DURATION_SCHEMA: Final[dict[str, pl.DataType]] = {
    SCALE_TYPE_COLUMN: pl.String(),
    HAND_COLUMN: pl.String(),
    N_COLUMN: pl.Int64(),
    BASE_DURATION_COLUMN: pl.String(),
    COUNT_COLUMN: pl.Int64(),
}

type BaseDurationCountsByGroup = dict[tuple[ScaleType, Hand, int], Counter[Fraction]]


def figure_counts_frame(
    counts_by_scale: FigureNGramCountsByScale,
    *,
    limit_per_group: int | None = None,
) -> pl.DataFrame:
    if limit_per_group is not None and limit_per_group <= 0:
        raise ValueError("limit_per_group must be positive")

    records: list[dict[str, str | int]] = []
    for scale_type, counts_by_hand in sorted(counts_by_scale.items(), key=lambda item: item[0].value):
        for hand, counts_by_n in sorted(counts_by_hand.items(), key=lambda item: item[0].value):
            for n, figure_counts in sorted(counts_by_n.items()):
                for figure, count in figure_counts.most_common(limit_per_group):
                    records.append(
                        {
                            SCALE_TYPE_COLUMN: scale_type.value,
                            HAND_COLUMN: hand.value,
                            N_COLUMN: n,
                            COUNT_COLUMN: count,
                            FIGURE_COLUMN: figure.model_dump_json(),
                        }
                    )

    return pl.DataFrame(records, schema=FIGURE_COUNT_SCHEMA, orient="row")


def write_figure_counts(counts: FigureNGramCountsByScale, path: Path) -> None:
    write_table(figure_counts_frame(counts), path)


def read_figure_counts(path: Path) -> FigureNGramCountsByScale:
    return _figure_counts_from_frame(read_table(path))


def read_figure_counts_for_groups(
    path: Path,
    *,
    scale_type: ScaleType,
    groups: frozenset[tuple[Hand, int]],
) -> FigureNGramCountsByScale:
    allowed_hands = {hand.value for hand, _ in groups}
    allowed_lengths = {n for _, n in groups}
    frame = read_table(path).filter(
        (pl.col(SCALE_TYPE_COLUMN) == scale_type.value)
        & pl.col(HAND_COLUMN).is_in(allowed_hands)
        & pl.col(N_COLUMN).is_in(allowed_lengths)
    )
    return _figure_counts_from_frame(frame, allowed_groups=groups)


def read_base_duration_counts(path: Path) -> BaseDurationCountsByGroup:
    counts: BaseDurationCountsByGroup = {}
    for row in read_table(path).iter_rows(named=True):
        scale_type = ScaleType(row[SCALE_TYPE_COLUMN])
        hand = Hand(row[HAND_COLUMN])
        n = int(row[N_COLUMN])
        base_duration = parse_ratio(row[BASE_DURATION_COLUMN])
        counts.setdefault((scale_type, hand, n), Counter())[base_duration] += int(row[COUNT_COLUMN])

    return counts


def _figure_counts_from_frame(
    frame: pl.DataFrame,
    *,
    allowed_groups: frozenset[tuple[Hand, int]] | None = None,
) -> FigureNGramCountsByScale:
    counts: FigureNGramCountsByScale = {}
    for row in frame.iter_rows(named=True):
        hand = Hand(row[HAND_COLUMN])
        n = int(row[N_COLUMN])
        if allowed_groups is not None and (hand, n) not in allowed_groups:
            continue

        scale_type = ScaleType(row[SCALE_TYPE_COLUMN])
        figure = FigureNGram.model_validate_json(row[FIGURE_COLUMN])
        counts.setdefault(scale_type, {}).setdefault(hand, {}).setdefault(n, Counter())[figure] += int(
            row[COUNT_COLUMN]
        )

    return counts


def write_figure_profile(profile: FigureProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=JSON_INDENT), encoding="utf-8")


def read_figure_profile(path: Path) -> FigureProfile:
    return FigureProfile.model_validate_json(path.read_text(encoding="utf-8"))


def write_figure_sample_counts_jsonl(
    sample_counts: Sequence[FigureSampleCounts],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample_count in sample_counts:
            file.write(sample_count.model_dump_json())
            file.write("\n")


def read_figure_sample_counts_jsonl(path: Path) -> list[FigureSampleCounts]:
    if not path.exists():
        return []

    return [
        FigureSampleCounts.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
