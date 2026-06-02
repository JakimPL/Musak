from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Self

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor

from musak_model.harmony.decoding.schema import ChordWindow
from musak_model.harmony.schema import Chord
from musak_shared.elements import HARMONIC_FUNCTION_BY_DEGREE, HarmonicFunction


class HarmonicPlanWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: Fraction
    end: Fraction
    chord: Chord

    @property
    def harmonic_function(self) -> HarmonicFunction | None:
        return harmonic_function_for_chord(self.chord)

    @classmethod
    def from_chord_window(cls, chord_window: ChordWindow) -> HarmonicPlanWindow:
        return cls(start=chord_window.start, end=chord_window.end, chord=chord_window.chord)

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.end <= self.start:
            raise ValueError("harmonic plan window end must be greater than start")

        return self


@dataclass(frozen=True)
class HarmonicPlanIds:
    harmonic_function_id: int
    root_degree_id: int
    root_accidental_id: int
    quality_id: int
    extension_id: int
    chord_change_id: int


@dataclass(frozen=True)
class HarmonicPlanInputTensors:
    harmonic_function_ids: Tensor
    root_degree_ids: Tensor
    root_accidental_ids: Tensor
    quality_ids: Tensor
    extension_ids: Tensor
    chord_change_ids: Tensor

    def to(self, device: torch.device) -> HarmonicPlanInputTensors:
        return HarmonicPlanInputTensors(
            harmonic_function_ids=self.harmonic_function_ids.to(device),
            root_degree_ids=self.root_degree_ids.to(device),
            root_accidental_ids=self.root_accidental_ids.to(device),
            quality_ids=self.quality_ids.to(device),
            extension_ids=self.extension_ids.to(device),
            chord_change_ids=self.chord_change_ids.to(device),
        )

    @property
    def shape(self) -> torch.Size:
        return self.harmonic_function_ids.shape


def harmonic_function_for_chord(chord: Chord) -> HarmonicFunction | None:
    return HARMONIC_FUNCTION_BY_DEGREE.get(chord.root_degree)


def harmonic_plan_windows_from_chord_windows(
    chord_windows: Sequence[ChordWindow],
) -> tuple[HarmonicPlanWindow, ...]:
    return tuple(HarmonicPlanWindow.from_chord_window(chord_window) for chord_window in chord_windows)
