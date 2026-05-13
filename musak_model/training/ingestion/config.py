from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.files import load_yaml_config
from musak_model.data.config import SegmentationConfig
from musak_model.paths import CONFIGS_DIR

_DEFAULT_SPLIT_SEED: Final[int] = 17
INGESTION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "ingestion.yml"


class IngestionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    segmentation: SegmentationConfig
    validation_fraction: float = Field(ge=0, lt=1)
    split_seed: int = _DEFAULT_SPLIT_SEED
    difficulty_labels: dict[str, int] | None = None

    @classmethod
    def load(cls, path: Path = INGESTION_CONFIG_PATH) -> IngestionConfig:
        parsed = load_yaml_config(path)
        parsed["segmentation"] = SegmentationConfig.model_validate(parsed["segmentation"])
        return cls.model_validate(parsed)
