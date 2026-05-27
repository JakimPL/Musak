from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from musak_model.tokens.schema import Hand, ScaleType

RHYTHM_DIR_NAME: Final[str] = "rhythm"
RHYTHM_PROFILE_NAME: Final[str] = "profile.json"
RHYTHM_COUNTS_NAME: Final[str] = "counts.csv"

RHYTHM_SCALE_TYPE_COLUMN: Final[str] = "scale_type"
RHYTHM_TIME_SIGNATURE_COLUMN: Final[str] = "time_signature"
RHYTHM_HAND_COLUMN: Final[str] = "hand"
RHYTHM_KIND_COLUMN: Final[str] = "kind"
RHYTHM_PARAMETER_COLUMN: Final[str] = "parameter"
RHYTHM_VALUE_COLUMN: Final[str] = "value"
RHYTHM_COUNT_COLUMN: Final[str] = "count"
RHYTHM_COUNT_CSV_COLUMNS: Final[tuple[str, ...]] = (
    RHYTHM_SCALE_TYPE_COLUMN,
    RHYTHM_TIME_SIGNATURE_COLUMN,
    RHYTHM_HAND_COLUMN,
    RHYTHM_KIND_COLUMN,
    RHYTHM_PARAMETER_COLUMN,
    RHYTHM_VALUE_COLUMN,
    RHYTHM_COUNT_COLUMN,
)

type RhythmMetricKind = Literal[
    "rhythm_ngram",
    "duration_value",
    "onset_grid_alignment",
    "duration_grid_alignment",
    "strong_beat_onset",
]


class RhythmGroupKey(NamedTuple):
    scale_type: str
    time_signature: str
    hand: str
    kind: RhythmMetricKind
    parameter: str


class RhythmCountKey(NamedTuple):
    scale_type: str
    time_signature: str
    hand: str
    kind: RhythmMetricKind
    parameter: str
    value: str


type RhythmCountCounter = Counter[RhythmCountKey]


@dataclass(frozen=True)
class RhythmArtifactPaths:
    root_directory: Path
    profile_path: Path
    counts_path: Path


class RhythmProfileMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    rhythm_min_n: int = Field(gt=0)
    rhythm_max_n: int = Field(gt=0)
    grid_alignment_denominators: tuple[int, ...]
    strong_beat_offsets: tuple[Fraction, ...]
    sample_count: int = Field(ge=0)


class RhythmProfileGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scale_type: ScaleType
    time_signature: str
    hand: Hand
    kind: RhythmMetricKind
    parameter: str
    total: int = Field(ge=0)
    unique_values: int = Field(ge=0)


class RhythmProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: RhythmProfileMetadata
    groups: tuple[RhythmProfileGroup, ...]


def rhythm_artifact_paths_for_figure_root(figure_root_directory: Path) -> RhythmArtifactPaths:
    root_directory = figure_root_directory / RHYTHM_DIR_NAME
    return RhythmArtifactPaths(
        root_directory=root_directory,
        profile_path=root_directory / RHYTHM_PROFILE_NAME,
        counts_path=root_directory / RHYTHM_COUNTS_NAME,
    )
