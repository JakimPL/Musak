from dataclasses import dataclass
from fractions import Fraction

from torch import Tensor

from musak_model.auxiliary.schema import MusicalAuxiliaryTargetTensors
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.training.dataset.factorized import TokenAttributeTargetTensors


@dataclass(frozen=True)
class TrainingExample:
    input_token_ids: Tensor
    target_token_ids: Tensor
    target_token_attributes: TokenAttributeTargetTensors
    musical_auxiliary_targets: MusicalAuxiliaryTargetTensors
    target_bar_positions: Tensor
    bar_positions: Tensor
    bar_relative_ticks: Tensor
    bar_duration_ticks: Tensor
    active_hand_ids: Tensor
    harmonic_plan: HarmonicPlanInputTensors | None
    structural_control_ids: Tensor
    scale_root: int
    scale_type_id: int
    time_numerator: int
    time_denominator: int
    bar_count: int
    bar_durations: tuple[Fraction, ...] | None
    difficulty_id: int | None
    conditioning_scale_type_id: int
    conditioning_time_signature_id: int


@dataclass(frozen=True)
class TrainingBatch:
    input_token_ids: Tensor
    target_token_ids: Tensor
    target_token_attributes: TokenAttributeTargetTensors
    musical_auxiliary_targets: MusicalAuxiliaryTargetTensors
    target_bar_positions: Tensor
    bar_positions: Tensor
    bar_relative_ticks: Tensor
    bar_duration_ticks: Tensor
    active_hand_ids: Tensor
    harmonic_plan: HarmonicPlanInputTensors | None
    structural_control_ids: Tensor
    scale_roots: Tensor
    scale_type_ids: Tensor
    time_numerators: Tensor
    time_denominators: Tensor
    bar_counts: Tensor
    bar_durations: tuple[tuple[Fraction, ...] | None, ...]
    token_padding_mask: Tensor
    difficulty_ids: Tensor | None
    conditioning_scale_type_ids: Tensor
    conditioning_time_signature_ids: Tensor
