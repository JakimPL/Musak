from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from musak_model.paths import CONFIGS_DIR

_DEFAULT_SPLIT_SEED: Final[int] = 17
CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "ingestion.yml"


class IngestionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_bars: int
    stride_bars: int
    validation_fraction: float
    split_seed: int = _DEFAULT_SPLIT_SEED
    difficulty_labels: dict[str, int] | None = None

    @field_validator("window_bars", "stride_bars")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("window_bars and stride_bars must be > 0")

        return value

    @field_validator("validation_fraction")
    @classmethod
    def _validate_fraction(cls, value: float) -> float:
        if not 0 <= value < 1:
            raise ValueError("validation_fraction must be in [0, 1)")

        return value

    @classmethod
    def load_config(cls, path: Path = CONFIG_PATH) -> IngestionConfig:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"expected mapping in config file: {path}")

        return cls.model_validate(parsed)
