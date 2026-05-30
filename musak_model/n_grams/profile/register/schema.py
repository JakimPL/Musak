from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

REGISTER_DIR_NAME: Final[str] = "register"
REGISTER_METADATA_NAME: Final[str] = "metadata.json"
REGISTER_STATISTICS_NAME: Final[str] = "statistics.parquet"

REGISTER_SCALE_TYPE_COLUMN: Final[str] = "scale_type"
REGISTER_HAND_COLUMN: Final[str] = "hand"
REGISTER_TREND_SQUARE_SUM_COLUMN: Final[str] = "trend_square_sum"
REGISTER_RESIDUAL_SQUARE_SUM_COLUMN: Final[str] = "residual_square_sum"
REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN: Final[str] = "residual_lag_product_sum"
REGISTER_ELEMENT_COUNT_COLUMN: Final[str] = "element_count"

REGISTER_STATISTICS_SCHEMA: Final[dict[str, pl.DataType]] = {
    REGISTER_SCALE_TYPE_COLUMN: pl.String(),
    REGISTER_HAND_COLUMN: pl.String(),
    REGISTER_TREND_SQUARE_SUM_COLUMN: pl.Float64(),
    REGISTER_RESIDUAL_SQUARE_SUM_COLUMN: pl.Float64(),
    REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN: pl.Float64(),
    REGISTER_ELEMENT_COUNT_COLUMN: pl.Int64(),
}


class RegisterStatisticsKey(NamedTuple):
    scale_type: str
    hand: str


@dataclass(frozen=True)
class RegisterSums:
    trend_square_sum: float
    residual_square_sum: float
    residual_lag_product_sum: float
    element_count: int


type RegisterStatistics = dict[RegisterStatisticsKey, RegisterSums]


@dataclass(frozen=True)
class RegisterArtifactPaths:
    root_directory: Path
    metadata_path: Path
    statistics_path: Path


class RegisterProfileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arch_basis_count: int = Field(gt=0)
    sample_count: int = Field(ge=0)


def register_artifact_paths_for_figure_root(figure_root_directory: Path) -> RegisterArtifactPaths:
    root_directory = figure_root_directory / REGISTER_DIR_NAME
    return RegisterArtifactPaths(
        root_directory=root_directory,
        metadata_path=root_directory / REGISTER_METADATA_NAME,
        statistics_path=root_directory / REGISTER_STATISTICS_NAME,
    )
