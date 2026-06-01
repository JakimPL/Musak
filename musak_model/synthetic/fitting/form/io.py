from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from musak_model.synthetic.fitting.form.statistics import (
    ClosingCounts,
    ClosingKey,
    FormStatistics,
    HistogramCounts,
    HistogramKey,
    PhraseLengthCounts,
    PhraseLengthKey,
    SegmentLengthCounts,
    SegmentLengthKey,
)
from musak_shared.tables import read_table, write_table

FORM_DIR_NAME: Final[str] = "form"
FORM_DATABASE_NAME: Final[str] = "form.sqlite3"
PHRASE_LENGTHS_NAME: Final[str] = "phrase_lengths.parquet"
SEGMENT_LENGTHS_NAME: Final[str] = "segment_lengths.parquet"
CLOSINGS_NAME: Final[str] = "closings.parquet"
SIMILARITY_HISTOGRAM_NAME: Final[str] = "similarity_histogram.parquet"
BEST_MATCH_HISTOGRAM_NAME: Final[str] = "best_match_histogram.parquet"

SCALE_TYPE_COLUMN: Final[str] = "scale_type"
PHRASE_LENGTH_BARS_COLUMN: Final[str] = "phrase_length_bars"
SEGMENT_LENGTH_BARS_COLUMN: Final[str] = "segment_length_bars"
IS_FINAL_COLUMN: Final[str] = "is_final"
FUNCTIONS_COLUMN: Final[str] = "functions"
BUCKET_COLUMN: Final[str] = "bucket"
COUNT_COLUMN: Final[str] = "count"

_PHRASE_LENGTHS_SCHEMA: Final[dict[str, pl.DataType]] = {
    SCALE_TYPE_COLUMN: pl.String(),
    PHRASE_LENGTH_BARS_COLUMN: pl.Int64(),
    COUNT_COLUMN: pl.Int64(),
}
_SEGMENT_LENGTHS_SCHEMA: Final[dict[str, pl.DataType]] = {
    SCALE_TYPE_COLUMN: pl.String(),
    SEGMENT_LENGTH_BARS_COLUMN: pl.Int64(),
    COUNT_COLUMN: pl.Int64(),
}
_CLOSINGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    SCALE_TYPE_COLUMN: pl.String(),
    IS_FINAL_COLUMN: pl.Boolean(),
    FUNCTIONS_COLUMN: pl.String(),
    COUNT_COLUMN: pl.Int64(),
}
_HISTOGRAM_SCHEMA: Final[dict[str, pl.DataType]] = {
    SCALE_TYPE_COLUMN: pl.String(),
    BUCKET_COLUMN: pl.Int64(),
    COUNT_COLUMN: pl.Int64(),
}


@dataclass(frozen=True)
class FormArtifactPaths:
    root_directory: Path
    database_path: Path
    phrase_lengths_path: Path
    segment_lengths_path: Path
    closings_path: Path
    similarity_histogram_path: Path
    best_match_histogram_path: Path


def form_artifact_paths_for_figure_root(figure_root_directory: Path) -> FormArtifactPaths:
    root_directory = figure_root_directory / FORM_DIR_NAME
    return FormArtifactPaths(
        root_directory=root_directory,
        database_path=root_directory / FORM_DATABASE_NAME,
        phrase_lengths_path=root_directory / PHRASE_LENGTHS_NAME,
        segment_lengths_path=root_directory / SEGMENT_LENGTHS_NAME,
        closings_path=root_directory / CLOSINGS_NAME,
        similarity_histogram_path=root_directory / SIMILARITY_HISTOGRAM_NAME,
        best_match_histogram_path=root_directory / BEST_MATCH_HISTOGRAM_NAME,
    )


def write_phrase_length_counts(counts: PhraseLengthCounts, path: Path) -> None:
    records = [
        {
            SCALE_TYPE_COLUMN: key.scale_type,
            PHRASE_LENGTH_BARS_COLUMN: key.phrase_length_bars,
            COUNT_COLUMN: count,
        }
        for key, count in sorted(counts.items())
    ]
    write_table(pl.DataFrame(records, schema=_PHRASE_LENGTHS_SCHEMA, orient="row"), path)


def read_phrase_length_counts(path: Path) -> PhraseLengthCounts:
    counts: PhraseLengthCounts = Counter()
    for row in read_table(path).iter_rows(named=True):
        key = PhraseLengthKey(
            scale_type=row[SCALE_TYPE_COLUMN],
            phrase_length_bars=int(row[PHRASE_LENGTH_BARS_COLUMN]),
        )
        counts[key] += int(row[COUNT_COLUMN])

    return counts


def write_segment_length_counts(counts: SegmentLengthCounts, path: Path) -> None:
    records = [
        {
            SCALE_TYPE_COLUMN: key.scale_type,
            SEGMENT_LENGTH_BARS_COLUMN: key.segment_length_bars,
            COUNT_COLUMN: count,
        }
        for key, count in sorted(counts.items())
    ]
    write_table(pl.DataFrame(records, schema=_SEGMENT_LENGTHS_SCHEMA, orient="row"), path)


def read_segment_length_counts(path: Path) -> SegmentLengthCounts:
    counts: SegmentLengthCounts = Counter()
    for row in read_table(path).iter_rows(named=True):
        key = SegmentLengthKey(
            scale_type=row[SCALE_TYPE_COLUMN],
            segment_length_bars=int(
                row[SEGMENT_LENGTH_BARS_COLUMN],
            ),
        )
        counts[key] += int(row[COUNT_COLUMN])

    return counts


def write_closing_counts(counts: ClosingCounts, path: Path) -> None:
    records = [
        {
            SCALE_TYPE_COLUMN: key.scale_type,
            IS_FINAL_COLUMN: key.is_final,
            FUNCTIONS_COLUMN: key.functions,
            COUNT_COLUMN: count,
        }
        for key, count in sorted(counts.items())
    ]
    write_table(pl.DataFrame(records, schema=_CLOSINGS_SCHEMA, orient="row"), path)


def read_closing_counts(path: Path) -> ClosingCounts:
    counts: ClosingCounts = Counter()
    for row in read_table(path).iter_rows(named=True):
        key = ClosingKey(
            scale_type=row[SCALE_TYPE_COLUMN],
            is_final=bool(row[IS_FINAL_COLUMN]),
            functions=row[FUNCTIONS_COLUMN],
        )
        counts[key] += int(row[COUNT_COLUMN])

    return counts


def write_histogram_counts(counts: HistogramCounts, path: Path) -> None:
    records = [
        {
            SCALE_TYPE_COLUMN: key.scale_type,
            BUCKET_COLUMN: key.bucket,
            COUNT_COLUMN: count,
        }
        for key, count in sorted(counts.items())
    ]
    write_table(pl.DataFrame(records, schema=_HISTOGRAM_SCHEMA, orient="row"), path)


def read_histogram_counts(path: Path) -> HistogramCounts:
    counts: HistogramCounts = Counter()
    for row in read_table(path).iter_rows(named=True):
        key = HistogramKey(
            scale_type=row[SCALE_TYPE_COLUMN],
            bucket=int(row[BUCKET_COLUMN]),
        )
        counts[key] += int(row[COUNT_COLUMN])

    return counts


def read_form_statistics(paths: FormArtifactPaths) -> FormStatistics | None:
    if not paths.phrase_lengths_path.exists():
        return None

    return FormStatistics(
        phrase_length_counts=read_phrase_length_counts(paths.phrase_lengths_path),
        segment_length_counts=read_segment_length_counts(paths.segment_lengths_path),
        closing_counts=read_closing_counts(paths.closings_path),
        similarity_histogram=read_histogram_counts(paths.similarity_histogram_path),
        best_match_histogram=read_histogram_counts(paths.best_match_histogram_path),
    )
