from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_model.common.files import load_yaml_config
from musak_model.paths import CONFIGS_DIR

TOKENIZATION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "tokens" / "tokenization.yml"


class TokenizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    shortest_duration: int = Field(gt=0)
    max_tuplets: tuple[int, ...] = Field(default=(3,), min_length=1)
    max_dots: int = Field(default=1, ge=0)

    @field_validator("shortest_duration")
    @classmethod
    def _validate_shortest_duration_power_of_two(cls, value: int) -> int:
        if value & (value - 1) != 0:
            raise ValueError("shortest_duration must be a power of 2")

        return value

    @field_validator("max_tuplets")
    @classmethod
    def _validate_max_tuplets(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not all(tuplet > 1 for tuplet in value):
            raise ValueError("all tuplet divisors must be > 1")

        return value

    @property
    def shortest_duration_fraction(self) -> Fraction:
        return Fraction(1, self.shortest_duration)

    @classmethod
    def load(cls, path: Path = TOKENIZATION_CONFIG_PATH) -> TokenizationConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)
