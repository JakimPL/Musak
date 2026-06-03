from __future__ import annotations

from dataclasses import dataclass

from musak_model.synthetic.validation.config import SyntheticValidationConfig
from musak_model.tokens.schema import ScaleType


@dataclass
class MetricOptions:
    """A full `GenerationEvaluationOptions` for a single scale.

    The reused metric helpers read only the musical/constraint fields; the neural-sampling fields exist
    purely to satisfy the protocol and are never consulted.
    """

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
    enabled: bool = True
    every_epochs: int = 1
    soft_sample_count: int = 0
    hard_sample_count: int = 0
    max_new_tokens: int = 0
    seed: int = 0
    temperature: float = 1.0
    top_k: int | None = None


def metric_options(config: SyntheticValidationConfig, scale_type: ScaleType) -> MetricOptions:
    return MetricOptions(
        scale_root=config.scale_root,
        scale_type=scale_type,
        time_numerator=config.time_numerator,
        time_denominator=config.time_denominator,
        bar_count=config.bar_count,
        minimum_duration_denominator=config.minimum_duration_denominator,
        allow_dotted_durations=config.allow_dotted_durations,
        max_notes_per_hand=config.max_notes_per_hand,
        maximum_onset_span_semitones=config.maximum_onset_span_semitones,
        maximum_pitch_gap_semitones=config.maximum_pitch_gap_semitones,
        maximum_static_hand_span_degrees=config.maximum_static_hand_span_degrees,
    )
