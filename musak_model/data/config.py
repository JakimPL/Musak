from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import SEGMENTATION_CONFIG_PATH
from musak_shared.files import load_yaml_config

DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN: Final[float] = 0.08
DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN: Final[float] = 0.03
DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION: Final[float] = 0.10
DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT: Final[int] = 9


class SegmentationMode(StrEnum):
    WINDOWED = "windowed"
    WHOLE_FILE = "whole_file"


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window_bars: int = Field(gt=0)
    stride_bars: int = Field(gt=0)
    mode: SegmentationMode = SegmentationMode.WINDOWED

    @classmethod
    def load(cls, path: Path = SEGMENTATION_CONFIG_PATH) -> SegmentationConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)


def load_segmentation_config(
    path: Path,
    *,
    window_bars: int | None = None,
    stride_bars: int | None = None,
    mode: SegmentationMode | None = None,
) -> SegmentationConfig:
    config = SegmentationConfig.load(path)
    return config.model_copy(
        update={
            key: value
            for key, value in {
                "window_bars": window_bars,
                "stride_bars": stride_bars,
                "mode": mode,
            }.items()
            if value is not None
        }
    )


class DataProcessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    remove_segments_with_silent_bars: bool
    scale_match_support_score_margin: float = Field(DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN, ge=0, le=1)
    scale_match_selection_score_margin: float = Field(DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN, ge=0, le=1)
    scale_match_maximum_unexplained_weight_fraction: float = Field(
        DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
        ge=0,
        le=1,
    )
    scale_match_maximum_explanation_pitch_class_count: int = Field(
        DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
        ge=0,
        le=12,
    )


def load_difficulty_labels(path: Path | None) -> dict[str, int | None] | None:
    if path is None:
        return None

    parsed = load_yaml_config(path)
    labels: dict[str, int | None] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not (isinstance(value, int) or value is None):
            raise ValueError("difficulty labels must be a mapping of relative source path to integer or null")

        labels[key] = value

    return labels
