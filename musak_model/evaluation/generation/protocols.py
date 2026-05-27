from __future__ import annotations

from typing import Protocol

from torch import Tensor

from musak_model.tokens.schema import ScaleType


class GenerationEvaluationOptions(Protocol):
    enabled: bool
    every_epochs: int
    soft_sample_count: int
    hard_sample_count: int
    max_new_tokens: int
    seed: int
    temperature: float
    top_k: int | None
    scale_root: int
    scale_type: ScaleType
    time_numerator: int
    time_denominator: int
    bar_count: int
    minimum_duration_denominator: int | None
    allow_dotted_durations: bool
    max_notes_per_hand: int | None
    maximum_onset_span_semitones: int | None
    maximum_pitch_gap_semitones: int | None
    maximum_static_hand_span_degrees: int | None


class GenerationConditioningOptions(Protocol):
    use_time_signature: bool
    use_scale_type: bool
    use_structural_conditioning: bool


class GenerationModel(Protocol):
    def eval(self) -> GenerationModel: ...

    def train(self, mode: bool = True) -> GenerationModel: ...

    @property
    def training(self) -> bool: ...

    def __call__(
        self,
        token_ids: Tensor,
        *,
        bar_positions: Tensor,
        difficulty_ids: Tensor | None = None,
        scale_type_ids: Tensor | None = None,
        time_signature_ids: Tensor | None = None,
        structural_control_ids: Tensor | None = None,
        token_padding_mask: Tensor | None = None,
    ) -> Tensor: ...
