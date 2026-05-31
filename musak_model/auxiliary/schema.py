from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

MUSICAL_AUXILIARY_TARGET_IGNORE_ID: Final[int] = -1


@dataclass(frozen=True)
class MusicalAuxiliaryTargetIds:
    note_density_id: int
    rhythmic_diversity_id: int
    voice_independence_id: int
    uses_accidentals_id: int
    dotted_duration_id: int
    hand_span_id: int


@dataclass(frozen=True)
class MusicalBarAuxiliaryTargetTensors:
    note_density_ids: Tensor
    rhythmic_diversity_ids: Tensor
    voice_independence_ids: Tensor
    uses_accidentals_ids: Tensor
    dotted_duration_ids: Tensor
    hand_span_ids: Tensor

    def to(self, device: torch.device) -> MusicalBarAuxiliaryTargetTensors:
        return MusicalBarAuxiliaryTargetTensors(
            note_density_ids=self.note_density_ids.to(device),
            rhythmic_diversity_ids=self.rhythmic_diversity_ids.to(device),
            voice_independence_ids=self.voice_independence_ids.to(device),
            uses_accidentals_ids=self.uses_accidentals_ids.to(device),
            dotted_duration_ids=self.dotted_duration_ids.to(device),
            hand_span_ids=self.hand_span_ids.to(device),
        )


@dataclass(frozen=True)
class MusicalAuxiliaryTargetTensors:
    note_density_ids: Tensor
    rhythmic_diversity_ids: Tensor
    voice_independence_ids: Tensor
    uses_accidentals_ids: Tensor
    dotted_duration_ids: Tensor
    hand_span_ids: Tensor
    bar_targets: MusicalBarAuxiliaryTargetTensors

    def to(self, device: torch.device) -> MusicalAuxiliaryTargetTensors:
        return MusicalAuxiliaryTargetTensors(
            note_density_ids=self.note_density_ids.to(device),
            rhythmic_diversity_ids=self.rhythmic_diversity_ids.to(device),
            voice_independence_ids=self.voice_independence_ids.to(device),
            uses_accidentals_ids=self.uses_accidentals_ids.to(device),
            dotted_duration_ids=self.dotted_duration_ids.to(device),
            hand_span_ids=self.hand_span_ids.to(device),
            bar_targets=self.bar_targets.to(device),
        )


@dataclass(frozen=True)
class MusicalBarAuxiliaryLogits:
    note_density: Tensor
    rhythmic_diversity: Tensor
    voice_independence: Tensor
    uses_accidentals: Tensor
    dotted_duration: Tensor
    hand_span: Tensor


@dataclass(frozen=True)
class MusicalAuxiliaryLogits:
    note_density: Tensor
    rhythmic_diversity: Tensor
    voice_independence: Tensor
    uses_accidentals: Tensor
    dotted_duration: Tensor
    hand_span: Tensor
    bar: MusicalBarAuxiliaryLogits
