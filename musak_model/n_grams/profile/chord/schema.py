from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from musak_model.harmony.decoding.config import ChordDecoderConfig
from musak_model.harmony.schema import Chord
from musak_model.harmony.vocabulary import ChordVocabularyConfig

CHORD_DIR_NAME: Final[str] = "chord"
CHORD_METADATA_NAME: Final[str] = "metadata.json"
CHORD_TRANSITIONS_NAME: Final[str] = "transitions.parquet"
CHORD_FIGURE_NAME: Final[str] = "figure_by_chord.parquet"

CHORD_SOURCE_COLUMN: Final[str] = "source_chord"
CHORD_DESTINATION_COLUMN: Final[str] = "destination_chord"
CHORD_SCALE_TYPE_COLUMN: Final[str] = "scale_type"
CHORD_HAND_COLUMN: Final[str] = "hand"
CHORD_N_COLUMN: Final[str] = "n"
CHORD_CHORD_COLUMN: Final[str] = "chord"
CHORD_FIGURE_COLUMN: Final[str] = "figure"
CHORD_COUNT_COLUMN: Final[str] = "count"

INITIAL_CHORD_SOURCE: Final[str] = ""

CHORD_TRANSITIONS_SCHEMA: Final[dict[str, pl.DataType]] = {
    CHORD_SCALE_TYPE_COLUMN: pl.String(),
    CHORD_SOURCE_COLUMN: pl.String(),
    CHORD_DESTINATION_COLUMN: pl.String(),
    CHORD_COUNT_COLUMN: pl.Int64(),
}

CHORD_FIGURE_SCHEMA: Final[dict[str, pl.DataType]] = {
    CHORD_SCALE_TYPE_COLUMN: pl.String(),
    CHORD_HAND_COLUMN: pl.String(),
    CHORD_N_COLUMN: pl.Int64(),
    CHORD_CHORD_COLUMN: pl.String(),
    CHORD_FIGURE_COLUMN: pl.String(),
    CHORD_COUNT_COLUMN: pl.Int64(),
}


class ChordTransitionKey(NamedTuple):
    scale_type: str
    source_chord: str
    destination_chord: str


class FigureByChordCountKey(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int
    chord: str
    figure: str


type ChordTransitionCounts = Counter[ChordTransitionKey]
type FigureByChordCounts = Counter[FigureByChordCountKey]


@dataclass(frozen=True)
class ChordStatistics:
    transition_counts: ChordTransitionCounts
    figure_by_chord_counts: FigureByChordCounts


@dataclass(frozen=True)
class ChordDecodeSpec:
    decoder_config: ChordDecoderConfig
    vocabulary: ChordVocabularyConfig


@dataclass(frozen=True)
class ChordArtifactPaths:
    root_directory: Path
    metadata_path: Path
    transitions_path: Path
    figure_path: Path


class ChordProfileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resolution: int = Field(gt=0)
    self_transition_bias: float = Field(ge=0)
    non_chord_penalty: float = Field(ge=0)
    sample_count: int = Field(ge=0)


def chord_to_key(chord: Chord) -> str:
    return chord.model_dump_json()


def chord_from_key(key: str) -> Chord:
    return Chord.model_validate_json(key)


def chord_artifact_paths_for_figure_root(figure_root_directory: Path) -> ChordArtifactPaths:
    root_directory = figure_root_directory / CHORD_DIR_NAME
    return ChordArtifactPaths(
        root_directory=root_directory,
        metadata_path=root_directory / CHORD_METADATA_NAME,
        transitions_path=root_directory / CHORD_TRANSITIONS_NAME,
        figure_path=root_directory / CHORD_FIGURE_NAME,
    )
