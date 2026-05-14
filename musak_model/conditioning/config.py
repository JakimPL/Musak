from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.files import load_yaml_config
from musak_model.conditioning.time_signature import TimeSignatureVocabulary, TimeSignatureVocabularyConfig
from musak_model.paths import CONFIGS_DIR
from musak_model.tokens.schema import ScaleType

CONDITIONING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "conditioning" / "conditioning.yml"


class DifficultyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_level: int = Field(ge=0)


class ConditioningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    difficulty: DifficultyConfig
    time_signature: TimeSignatureVocabularyConfig
    cfg_dropout_probability: float = Field(ge=0.0, lt=1.0)

    @property
    def num_difficulty_levels(self) -> int:
        return self.difficulty.max_level + 1

    @property
    def num_scale_types(self) -> int:
        return len(ScaleType)

    @property
    def num_time_signatures(self) -> int:
        return TimeSignatureVocabulary(self.time_signature).vocab_size

    @classmethod
    def load(cls, path: Path = CONDITIONING_CONFIG_PATH) -> ConditioningConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
