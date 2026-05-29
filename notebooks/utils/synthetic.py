from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from numpy.random import default_rng

from musak_model.data.schema import Segment
from musak_model.generation.constraints import GenerationConstraintError, GenerationConstraints
from musak_model.synthetic.base_durations import BaseDurationDistribution, load_base_duration_distribution
from musak_model.synthetic.figures import FigureVocabulary, load_figure_vocabulary
from musak_model.synthetic.harmony.decoding.candidates import spellable_candidates
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler
from musak_model.synthetic.processes.chord_track import ChordTrackSampler, uniform_transition_model
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig, HandCouplingSampler
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.substitution import GenerationTrace, SegmentGenerator, SubstitutionConfig
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType
from notebooks.utils.model_output import segment_decode_error

_SOURCE_FILE = Path("synthetic")


@dataclass(frozen=True)
class SyntheticInputs:
    figure_vocabulary: FigureVocabulary
    base_duration_distribution: BaseDurationDistribution
    duration_vocabulary: DurationVocabulary


def load_synthetic_inputs(figure_directory: Path) -> SyntheticInputs:
    return SyntheticInputs(
        figure_vocabulary=load_figure_vocabulary(figure_directory),
        base_duration_distribution=load_base_duration_distribution(figure_directory),
        duration_vocabulary=DurationVocabulary(TokenizationConfig.load()),
    )


@dataclass(frozen=True)
class SyntheticGenerationRequest:
    scale_root: int
    scale_type: str
    time_numerator: int
    time_denominator: int
    grid_count_per_bar: int
    chord_resolution: int
    bar_count: int
    seed: int
    min_n: int
    max_n: int
    lambda_curve: float
    lambda_harm: float
    lambda_accent: float
    commonness_bias: float
    max_resample_retries: int
    arch_basis_count: int
    arch_amplitude: float
    arch_decay: float
    ou_theta: float
    ou_sigma: float
    baseline_logit: float
    metric_gain: float
    metric_exponent: float
    envelope_basis_count: int
    envelope_amplitude: float
    envelope_decay: float
    co_activity_strength: float
    activity_right: float
    activity_left: float
    sync_strength: float
    self_transition_bias: float
    use_constraints: bool
    minimum_duration: str
    allow_dotted: bool
    max_notes_per_hand: int | None
    max_onset_span: int | None
    max_gap: int | None
    max_span: int | None


@dataclass(frozen=True)
class SyntheticGeneratedOutput:
    segment: Segment | None
    trace: GenerationTrace
    duration_vocabulary: DurationVocabulary
    decode_error: str | None
    error: str | None
    status_message: str
    status_kind: Literal["success", "warn"]


def generate_synthetic_segment(
    inputs: SyntheticInputs,
    request: SyntheticGenerationRequest,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SyntheticGeneratedOutput:
    duration_vocabulary = inputs.duration_vocabulary
    scale_type = ScaleType(request.scale_type)
    chord_vocabulary = ChordVocabularyConfig.load()
    chords = tuple(candidate.chord for candidate in spellable_candidates(chord_vocabulary, scale_type=scale_type))
    generator = SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=request.lambda_curve,
            lambda_harm=request.lambda_harm,
            lambda_accent=request.lambda_accent,
            commonness_bias=request.commonness_bias,
            max_resample_retries=request.max_resample_retries,
        ),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig(
                arch_basis_count=request.arch_basis_count,
                arch_amplitude=request.arch_amplitude,
                arch_decay=request.arch_decay,
                ou_theta=request.ou_theta,
                ou_sigma=request.ou_sigma,
            )
        ),
        accent_field_sampler=AccentFieldSampler(
            config=AccentFieldConfig(
                baseline_logit=request.baseline_logit,
                metric_gain=request.metric_gain,
                metric_exponent=request.metric_exponent,
                envelope_basis_count=request.envelope_basis_count,
                envelope_amplitude=request.envelope_amplitude,
                envelope_decay=request.envelope_decay,
            )
        ),
        hand_coupling_sampler=HandCouplingSampler(
            config=HandCouplingConfig(
                co_activity_strength=request.co_activity_strength,
                activity_right=request.activity_right,
                activity_left=request.activity_left,
                sync_strength=request.sync_strength,
            )
        ),
        chord_track_sampler=ChordTrackSampler(
            model=uniform_transition_model(chords, self_transition_bias=request.self_transition_bias)
        ),
        chord_vocabulary=chord_vocabulary,
        figure_vocabulary=inputs.figure_vocabulary,
        base_duration_distribution=inputs.base_duration_distribution,
        duration_vocabulary=duration_vocabulary,
        figure_lengths=tuple(range(request.min_n, request.max_n + 1)),
    )
    try:
        result = generator.generate(
            bar_count=request.bar_count,
            time_numerator=request.time_numerator,
            time_denominator=request.time_denominator,
            grid_count_per_bar=request.grid_count_per_bar,
            chord_resolution=request.chord_resolution,
            scale_root=request.scale_root,
            scale_type=scale_type,
            constraints=_build_constraints(request, scale_type=scale_type),
            rng=default_rng(request.seed),
            source_file=_SOURCE_FILE,
            progress_callback=progress_callback,
        )
    except (GenerationConstraintError, ValueError) as exception:
        return _failure(duration_vocabulary, f"Generation failed: {exception}")

    segment = result.segment
    decode_error = segment_decode_error(segment, duration_vocabulary=duration_vocabulary)
    status_message = (
        f"Generated {segment.bar_count} bar(s), {len(segment.tokens)} tokens | decode error: {decode_error or '-'}"
    )
    return SyntheticGeneratedOutput(
        segment=segment,
        trace=result.trace,
        duration_vocabulary=duration_vocabulary,
        decode_error=decode_error,
        error=None,
        status_message=status_message,
        status_kind="success" if decode_error is None else "warn",
    )


def _build_constraints(request: SyntheticGenerationRequest, *, scale_type: ScaleType) -> GenerationConstraints:
    if not request.use_constraints:
        return GenerationConstraints(
            time_numerator=request.time_numerator,
            time_denominator=request.time_denominator,
            bar_count=request.bar_count,
        )

    return GenerationConstraints(
        time_numerator=request.time_numerator,
        time_denominator=request.time_denominator,
        bar_count=request.bar_count,
        minimum_duration=Fraction(request.minimum_duration) if request.minimum_duration != "None" else None,
        allow_dotted_durations=request.allow_dotted,
        max_notes_per_hand=request.max_notes_per_hand,
        maximum_onset_span_semitones=request.max_onset_span,
        maximum_pitch_gap_semitones=request.max_gap,
        maximum_static_hand_span_degrees=request.max_span,
        scale_root=request.scale_root,
        scale_type=scale_type,
    )


def _failure(duration_vocabulary: DurationVocabulary, message: str) -> SyntheticGeneratedOutput:
    return SyntheticGeneratedOutput(
        segment=None,
        trace=GenerationTrace(samples=(), grid_count_per_bar=1, bar_count=0),
        duration_vocabulary=duration_vocabulary,
        decode_error=None,
        error=message,
        status_message=message,
        status_kind="warn",
    )
