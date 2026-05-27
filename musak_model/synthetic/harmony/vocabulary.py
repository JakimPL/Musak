from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_model.paths import CHORD_VOCABULARY_CONFIG_PATH
from musak_model.synthetic.harmony.schema import ChordExtension, ChordQuality
from musak_shared.files import load_yaml_config

TRIAD_INTERVAL_COUNT: Final[int] = 3


class QualityDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    intervals: tuple[int, ...]
    enabled: bool

    @field_validator("intervals")
    @classmethod
    def _validate_triad_intervals(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != TRIAD_INTERVAL_COUNT:
            raise ValueError(f"quality intervals must describe a triad of {TRIAD_INTERVAL_COUNT} semitone offsets")

        if value[0] != 0:
            raise ValueError("the first quality interval must be the root offset (0)")

        return value


class ExtensionDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    members: int = Field(ge=TRIAD_INTERVAL_COUNT)
    alterations: dict[int, int] = {}
    enabled: bool

    @field_validator("alterations")
    @classmethod
    def _validate_alterations(cls, value: dict[int, int]) -> dict[int, int]:
        if any(member < 0 for member in value):
            raise ValueError("alteration member indices must be non-negative")

        return value


class ChordVocabularyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    qualities: dict[ChordQuality, QualityDefinition]
    extensions: dict[ChordExtension, ExtensionDefinition]

    def quality_definition(self, quality: ChordQuality) -> QualityDefinition:
        return self.qualities[quality]

    def extension_definition(self, extension: ChordExtension) -> ExtensionDefinition:
        return self.extensions[extension]

    def enabled_qualities(self) -> tuple[ChordQuality, ...]:
        return tuple(quality for quality, definition in self.qualities.items() if definition.enabled)

    def enabled_extensions(self) -> tuple[ChordExtension, ...]:
        return tuple(extension for extension, definition in self.extensions.items() if definition.enabled)

    @classmethod
    def load(cls, path: Path = CHORD_VOCABULARY_CONFIG_PATH) -> ChordVocabularyConfig:
        return cls.model_validate(load_yaml_config(path))
