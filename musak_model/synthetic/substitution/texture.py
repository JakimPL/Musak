from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.tokens.schema import Hand


class HandTexture(StrEnum):
    MELODIC = "melodic"
    BLOCK_CHORD = "block_chord"
    SUSTAINED_BASS = "sustained_bass"


class AccompanimentRhythm(StrEnum):
    BLOCK_PER_WINDOW = "block_per_window"
    ACCENT_GATED = "accent_gated"


class AccompanimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rhythm: AccompanimentRhythm
    max_chord_notes: int = Field(gt=0)


class HandTextureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    right: HandTexture
    left: HandTexture
    accompaniment: AccompanimentConfig

    def texture(self, hand: Hand) -> HandTexture:
        return self.right if hand == Hand.RIGHT else self.left


ALL_MELODIC_TEXTURE: Final[HandTextureConfig] = HandTextureConfig(
    right=HandTexture.MELODIC,
    left=HandTexture.MELODIC,
    accompaniment=AccompanimentConfig(rhythm=AccompanimentRhythm.BLOCK_PER_WINDOW, max_chord_notes=3),
)
