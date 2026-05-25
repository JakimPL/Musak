from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.conditioning.structural.config import StructuralConditioningConfig
from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary, TimeSignatureVocabularyConfig
from musak_model.paths import CONDITIONING_CONFIG_PATH
from musak_model.tokens.schema import ScaleType
from musak_shared.files import load_yaml_config


class DifficultyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_level: int = Field(ge=0)


class ConditioningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    difficulty: DifficultyConfig
    time_signature: TimeSignatureVocabularyConfig
    structural: StructuralConditioningConfig = StructuralConditioningConfig()
    cfg_dropout_probability: float = Field(ge=0.0, lt=1.0)

    @property
    def num_difficulty_levels(self) -> int:
        return self.difficulty.max_level + 1

    @property
    def num_scale_types(self) -> int:
        return len(ScaleType)

    @property
    def num_time_signatures(self) -> int:
        return TimeSignatureVocabulary(self.time_signature).vocabulary_size

    @property
    def structural_vocabulary_sizes(self) -> tuple[int, ...]:
        return StructuralControlVocabulary(self.structural).vocabulary_sizes

    @classmethod
    def load(cls, path: Path = CONDITIONING_CONFIG_PATH) -> ConditioningConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
