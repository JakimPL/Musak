from dataclasses import dataclass
from pathlib import Path
from typing import Final

FIGURE_DIR_NAME: Final[str] = "figure"
FIGURE_CONFIG_NAME: Final[str] = "config.yml"
FIGURE_ALL_DIR_NAME: Final[str] = "all"
FIGURE_PROFILE_NAME: Final[str] = "profile.json"
FIGURE_COUNTS_NAME: Final[str] = "counts.csv"
FIGURE_BY_SAMPLE_NAME: Final[str] = "by_sample.jsonl"


@dataclass(frozen=True)
class FigureArtifactPaths:
    root_dir: Path
    config_path: Path
    all_dir: Path
    profile_path: Path
    counts_path: Path
    by_sample_path: Path


def figure_artifact_paths(encoded_dir: Path) -> FigureArtifactPaths:
    figure_dir = encoded_dir / FIGURE_DIR_NAME
    all_dir = figure_dir / FIGURE_ALL_DIR_NAME
    return FigureArtifactPaths(
        root_dir=figure_dir,
        config_path=figure_dir / FIGURE_CONFIG_NAME,
        all_dir=all_dir,
        profile_path=all_dir / FIGURE_PROFILE_NAME,
        counts_path=all_dir / FIGURE_COUNTS_NAME,
        by_sample_path=figure_dir / FIGURE_BY_SAMPLE_NAME,
    )
