from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import SEGMENTATION_CONFIG_PATH
from musak_shared.files import load_yaml_config


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window_bars: int = Field(gt=0)
    stride_bars: int = Field(gt=0)

    @classmethod
    def load(cls, path: Path = SEGMENTATION_CONFIG_PATH) -> SegmentationConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)


def load_segmentation_config(
    path: Path,
    *,
    window_bars: int | None = None,
    stride_bars: int | None = None,
) -> SegmentationConfig:
    config = SegmentationConfig.load(path)
    return config.model_copy(
        update={
            key: value
            for key, value in {
                "window_bars": window_bars,
                "stride_bars": stride_bars,
            }.items()
            if value is not None
        }
    )


class DataProcessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    remove_segments_with_silent_bars: bool


def load_difficulty_labels(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None

    parsed = load_yaml_config(path)
    labels: dict[str, int] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise ValueError("difficulty labels must be a mapping of file stem to integer difficulty level")

        labels[key] = value

    return labels
