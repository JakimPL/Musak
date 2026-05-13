from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.files import load_yaml_config
from musak_model.paths import CONFIGS_DIR

SEGMENTATION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "data" / "segmentation.yml"


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_bars: int = Field(gt=0)
    stride_bars: int = Field(gt=0)

    @classmethod
    def load(cls, path: Path = SEGMENTATION_CONFIG_PATH) -> SegmentationConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
