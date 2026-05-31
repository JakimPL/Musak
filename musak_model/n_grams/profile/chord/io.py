from collections import Counter
from pathlib import Path

import polars as pl

from musak_model.n_grams.profile.chord.schema import (
    CHORD_CHORD_COLUMN,
    CHORD_COUNT_COLUMN,
    CHORD_DESTINATION_COLUMN,
    CHORD_FIGURE_COLUMN,
    CHORD_FIGURE_SCHEMA,
    CHORD_HAND_COLUMN,
    CHORD_N_COLUMN,
    CHORD_SCALE_TYPE_COLUMN,
    CHORD_SOURCE_COLUMN,
    CHORD_TRANSITIONS_SCHEMA,
    ChordProfileMetadata,
    ChordTransitionCounts,
    ChordTransitionKey,
    FigureByChordCountKey,
    FigureByChordCounts,
)
from musak_shared.files import JSON_INDENT
from musak_shared.tables import read_table, write_table


def write_chord_transitions(counts: ChordTransitionCounts, path: Path) -> None:
    records = [
        {
            CHORD_SCALE_TYPE_COLUMN: key.scale_type,
            CHORD_SOURCE_COLUMN: key.source_chord,
            CHORD_DESTINATION_COLUMN: key.destination_chord,
            CHORD_COUNT_COLUMN: count,
        }
        for key, count in sorted(counts.items())
    ]
    write_table(pl.DataFrame(records, schema=CHORD_TRANSITIONS_SCHEMA, orient="row"), path)


def read_chord_transitions(path: Path) -> ChordTransitionCounts:
    counts: ChordTransitionCounts = Counter()
    for row in read_table(path).iter_rows(named=True):
        key = ChordTransitionKey(
            scale_type=row[CHORD_SCALE_TYPE_COLUMN],
            source_chord=row[CHORD_SOURCE_COLUMN],
            destination_chord=row[CHORD_DESTINATION_COLUMN],
        )
        counts[key] += int(row[CHORD_COUNT_COLUMN])

    return counts


def write_figure_by_chord(counts: FigureByChordCounts, path: Path) -> None:
    records = [
        {
            CHORD_SCALE_TYPE_COLUMN: key.scale_type,
            CHORD_HAND_COLUMN: key.hand,
            CHORD_N_COLUMN: key.figure_length,
            CHORD_CHORD_COLUMN: key.chord,
            CHORD_FIGURE_COLUMN: key.figure,
            CHORD_COUNT_COLUMN: count,
        }
        for key, count in sorted(counts.items())
    ]
    write_table(pl.DataFrame(records, schema=CHORD_FIGURE_SCHEMA, orient="row"), path)


def read_figure_by_chord(path: Path) -> FigureByChordCounts:
    counts: FigureByChordCounts = Counter()
    for row in read_table(path).iter_rows(named=True):
        key = FigureByChordCountKey(
            scale_type=row[CHORD_SCALE_TYPE_COLUMN],
            hand=row[CHORD_HAND_COLUMN],
            figure_length=int(row[CHORD_N_COLUMN]),
            chord=row[CHORD_CHORD_COLUMN],
            figure=row[CHORD_FIGURE_COLUMN],
        )
        counts[key] += int(row[CHORD_COUNT_COLUMN])

    return counts


def write_chord_metadata(metadata: ChordProfileMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.model_dump_json(indent=JSON_INDENT), encoding="utf-8")


def read_chord_metadata(path: Path) -> ChordProfileMetadata:
    return ChordProfileMetadata.model_validate_json(path.read_text(encoding="utf-8"))
