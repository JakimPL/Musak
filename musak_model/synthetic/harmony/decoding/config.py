from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_model.paths import CHORD_DECODING_CONFIG_PATH
from musak_shared.files import load_yaml_config
from musak_shared.misc import is_power_of_two


class ChordDecoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resolution: int = Field(gt=0)
    self_transition_bias: float = Field(ge=0)
    non_chord_penalty: float = Field(ge=0)

    @field_validator("resolution")
    @classmethod
    def _validate_resolution_power_of_two(cls, value: int) -> int:
        if not is_power_of_two(value):
            raise ValueError("resolution must be a power of two note value (1 whole, 2 half, 4 quarter, ...)")

        return value

    @classmethod
    def load(cls, path: Path = CHORD_DECODING_CONFIG_PATH) -> ChordDecoderConfig:
        return cls.model_validate(load_yaml_config(path))
