from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_model.common.files import load_yaml_config
from musak_model.paths import TOKENIZATION_CONFIG_PATH
from musak_shared.common import is_power_of_two


class TokenizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    shortest_duration: int = Field(gt=0)
    allowed_tuplets: tuple[int, ...]
    max_dots: int = Field(ge=0)

    @field_validator("shortest_duration")
    @classmethod
    def _validate_shortest_duration_power_of_two(cls, value: int) -> int:
        if not is_power_of_two(value):
            raise ValueError("shortest_duration must be a power of 2")

        return value

    @field_validator("allowed_tuplets")
    @classmethod
    def _validate_allowed_tuplets(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not all(tuplet > 1 for tuplet in value):
            raise ValueError("all allowed tuplet divisors must be > 1")

        return value

    @property
    def shortest_duration_fraction(self) -> Fraction:
        return Fraction(1, self.shortest_duration)

    @classmethod
    def load(cls, path: Path = TOKENIZATION_CONFIG_PATH) -> TokenizationConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
