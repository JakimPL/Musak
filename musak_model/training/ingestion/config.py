from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.files import load_yaml_config
from musak_model.paths import INGESTION_CONFIG_PATH

_DEFAULT_SPLIT_SEED: Final[int] = 17


class IngestionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    validation_fraction: float = Field(ge=0, lt=1)
    split_seed: int = _DEFAULT_SPLIT_SEED
    difficulty_labels: dict[str, int] | None = None
    processed_root: Path | None = None

    @classmethod
    def load(cls, path: Path = INGESTION_CONFIG_PATH) -> IngestionConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
