from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.files import load_yaml_config
from musak_model.paths import SEGMENTATION_CONFIG_PATH


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window_bars: int = Field(gt=0)
    stride_bars: int = Field(gt=0)

    @classmethod
    def load(cls, path: Path = SEGMENTATION_CONFIG_PATH) -> SegmentationConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
