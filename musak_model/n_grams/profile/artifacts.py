from dataclasses import dataclass
from pathlib import Path
from typing import Final

FIGURE_DIR_NAME: Final[str] = "figure"
FIGURE_CONFIG_NAME: Final[str] = "config.yml"
FIGURE_ALL_DIR_NAME: Final[str] = "all"
FIGURE_PROFILE_NAME: Final[str] = "profile.json"
FIGURE_COUNTS_NAME: Final[str] = "counts.parquet"
FIGURE_BASE_DURATIONS_NAME: Final[str] = "base_durations.parquet"
FIGURE_BY_SAMPLE_NAME: Final[str] = "by_sample.jsonl"


@dataclass(frozen=True)
class FigureArtifactPaths:
    root_directory: Path
    config_path: Path
    all_directory: Path
    profile_path: Path
    counts_path: Path
    base_durations_path: Path
    by_sample_path: Path


def figure_artifact_paths_from_root(figure_directory: Path) -> FigureArtifactPaths:
    all_directory = figure_directory / FIGURE_ALL_DIR_NAME
    return FigureArtifactPaths(
        root_directory=figure_directory,
        config_path=figure_directory / FIGURE_CONFIG_NAME,
        all_directory=all_directory,
        profile_path=all_directory / FIGURE_PROFILE_NAME,
        counts_path=all_directory / FIGURE_COUNTS_NAME,
        base_durations_path=all_directory / FIGURE_BASE_DURATIONS_NAME,
        by_sample_path=figure_directory / FIGURE_BY_SAMPLE_NAME,
    )


def figure_artifact_paths(encoded_directory: Path) -> FigureArtifactPaths:
    return figure_artifact_paths_from_root(encoded_directory / FIGURE_DIR_NAME)
