from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from musak_model.paths import N_GRAM_ANALYSIS_CONFIG_PATH
from musak_shared.files import load_yaml_config


class NGramAnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    limit_per_group: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_n_range(self) -> NGramAnalysisConfig:
        if self.max_n < self.min_n:
            raise ValueError("max_n must be greater than or equal to min_n")

        return self

    @classmethod
    def load(cls, path: Path = N_GRAM_ANALYSIS_CONFIG_PATH) -> NGramAnalysisConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
