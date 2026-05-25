import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.analysis.n_grams.figure.encoded import FigureNGramCountsByScale
from musak_model.analysis.n_grams.profile.schema import FigureProfile
from musak_model.processing.io import JSON_INDENT

SCALE_TYPE_COLUMN: Final[str] = "scale_type"
HAND_COLUMN: Final[str] = "hand"
N_COLUMN: Final[str] = "n"
COUNT_COLUMN: Final[str] = "count"
FIGURE_COLUMN: Final[str] = "figure"
COUNT_CSV_COLUMNS: Final[tuple[str, ...]] = (
    SCALE_TYPE_COLUMN,
    HAND_COLUMN,
    N_COLUMN,
    COUNT_COLUMN,
    FIGURE_COLUMN,
)

type FigureNGramCountRecord = dict[str, str | int]


def figure_count_records(
    counts_by_scale: FigureNGramCountsByScale,
    *,
    limit_per_group: int | None = None,
) -> list[FigureNGramCountRecord]:
    if limit_per_group is not None and limit_per_group <= 0:
        raise ValueError("limit_per_group must be positive")

    records: list[FigureNGramCountRecord] = []
    for scale_type, counts_by_hand in sorted(counts_by_scale.items(), key=lambda item: item[0].value):
        for hand, counts_by_n in sorted(counts_by_hand.items(), key=lambda item: item[0].value):
            for n, figure_counts in sorted(counts_by_n.items()):
                common_figures = figure_counts.most_common(limit_per_group)
                for figure, count in common_figures:
                    records.append(
                        {
                            SCALE_TYPE_COLUMN: scale_type.value,
                            HAND_COLUMN: hand.value,
                            N_COLUMN: n,
                            COUNT_COLUMN: count,
                            FIGURE_COLUMN: figure.model_dump_json(),
                        }
                    )

    return records


def write_figure_count_csv(
    records: Sequence[FigureNGramCountRecord],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COUNT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def write_figure_counts_csv(
    counts: FigureNGramCountsByScale,
    path: Path,
) -> None:
    write_figure_count_csv(figure_count_records(counts), path)


def write_figure_profile(profile: FigureProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=JSON_INDENT), encoding="utf-8")


def read_figure_profile(path: Path) -> FigureProfile:
    return FigureProfile.model_validate_json(path.read_text(encoding="utf-8"))
