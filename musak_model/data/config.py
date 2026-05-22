from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import SEGMENTATION_CONFIG_PATH
from musak_shared.files import load_yaml_config


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
