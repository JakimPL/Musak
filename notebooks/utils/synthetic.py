from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from numpy.random import default_rng

from musak_model.data.schema import Segment
from musak_model.generation.constraints import GenerationConstraintError, GenerationConstraints
from musak_model.harmony.decoding.candidates import spellable_candidates
from musak_model.harmony.schema import Chord
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.base_durations import BaseDurationDistribution, load_base_duration_distribution
from musak_model.synthetic.builder import build_segment_generator
from musak_model.synthetic.figures import (
    AnchoredFigureVocabulary,
    FigureVocabulary,
    load_anchored_figure_vocabulary,
    load_figure_vocabulary,
)
from musak_model.synthetic.fitting.artifacts import FittedGeneratorConfig, resolve_fitted_generator_config_path
from musak_model.synthetic.fitting.figure_by_chord import load_figure_by_chord_model
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.synthetic.processes.chord_track import (
    ChordTransitionModel,
    functional_transition_model,
    uniform_transition_model,
)
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig
from musak_model.synthetic.processes.pitch import RegisterCurveConfig
from musak_model.synthetic.substitution import (
    AccompanimentConfig,
    AccompanimentRhythm,
    FigureByChordModel,
    GenerationTrace,
    HandTexture,
    HandTextureConfig,
    SubstitutionConfig,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType
from notebooks.utils.model_output import segment_decode_error

_SOURCE_FILE = Path("synthetic")


@dataclass(frozen=True)
class SyntheticInputs:
    figure_vocabulary: FigureVocabulary
    anchored_figure_vocabulary: AnchoredFigureVocabulary
    base_duration_distribution: BaseDurationDistribution
    duration_vocabulary: DurationVocabulary
    fitted: FittedGeneratorConfig
    figure_by_chord_model: FigureByChordModel


def load_synthetic_inputs(figure_directory: Path) -> SyntheticInputs:
    return SyntheticInputs(
        figure_vocabulary=load_figure_vocabulary(figure_directory),
        anchored_figure_vocabulary=load_anchored_figure_vocabulary(figure_directory),
        base_duration_distribution=load_base_duration_distribution(figure_directory),
        duration_vocabulary=DurationVocabulary(TokenizationConfig.load()),
        fitted=_load_fitted_generator_config(figure_directory),
        figure_by_chord_model=load_figure_by_chord_model(figure_directory),
    )


def _load_fitted_generator_config(figure_directory: Path) -> FittedGeneratorConfig:
    path = resolve_fitted_generator_config_path(figure_directory)
    return FittedGeneratorConfig.read(path) if path is not None else FittedGeneratorConfig()


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
    monophonic: bool
    lambda_curve: float
    lambda_harmonic: float
    lambda_accent: float
    lambda_chord_figure: float
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
    functional_strength: float
    chord_model: str
    right_texture: str
    left_texture: str
    accompaniment_rhythm: str
    accompaniment_max_notes: int
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
    chord_transition_model, chord_model_warning = _chord_transition_model(
        request, inputs, chords=chords, scale_type=scale_type
    )
    setup_warnings = _setup_warnings(request, inputs, chord_model_warning=chord_model_warning)
    generator = build_segment_generator(
        substitution_config=SubstitutionConfig(
            lambda_curve=request.lambda_curve,
            lambda_harmonic=request.lambda_harmonic,
            lambda_accent=request.lambda_accent,
            lambda_chord_figure=request.lambda_chord_figure,
            commonness_bias=request.commonness_bias,
            max_resample_retries=request.max_resample_retries,
            monophonic=request.monophonic,
            texture=_hand_texture_config(request),
        ),
        register_curve_config=RegisterCurveConfig(
            arch_basis_count=request.arch_basis_count,
            arch_amplitude=request.arch_amplitude,
            arch_decay=request.arch_decay,
            ou_theta=request.ou_theta,
            ou_sigma=request.ou_sigma,
        ),
        register_curve_overrides=inputs.fitted.register_overrides,
        accent_field_config=AccentFieldConfig(
            baseline_logit=request.baseline_logit,
            metric_gain=request.metric_gain,
            metric_exponent=request.metric_exponent,
            envelope_basis_count=request.envelope_basis_count,
            envelope_amplitude=request.envelope_amplitude,
            envelope_decay=request.envelope_decay,
        ),
        accent_field_overrides=inputs.fitted.accent_overrides,
        hand_coupling_config=HandCouplingConfig(
            co_activity_strength=request.co_activity_strength,
            activity_right=request.activity_right,
            activity_left=request.activity_left,
            sync_strength=request.sync_strength,
        ),
        chord_transition_model=chord_transition_model,
        chord_vocabulary=chord_vocabulary,
        figure_vocabulary=inputs.figure_vocabulary,
        anchored_figure_vocabulary=inputs.anchored_figure_vocabulary,
        figure_by_chord_model=inputs.figure_by_chord_model,
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
        return _failure(duration_vocabulary, f"Generation failed: {exception}", warnings=setup_warnings)

    segment = result.segment
    decode_error = segment_decode_error(segment, duration_vocabulary=duration_vocabulary)
    base_status = (
        f"Generated {segment.bar_count} bar(s), {len(segment.tokens)} tokens | decode error: {decode_error or '-'}"
    )
    return SyntheticGeneratedOutput(
        segment=segment,
        trace=result.trace,
        duration_vocabulary=duration_vocabulary,
        decode_error=decode_error,
        error=None,
        status_message=_with_warnings(base_status, setup_warnings),
        status_kind="success" if decode_error is None and not setup_warnings else "warn",
    )


def _setup_warnings(
    request: SyntheticGenerationRequest,
    inputs: SyntheticInputs,
    *,
    chord_model_warning: str | None,
) -> list[str]:
    warnings: list[str] = []
    if chord_model_warning is not None:
        warnings.append(chord_model_warning)

    if request.lambda_chord_figure > 0.0 and not inputs.figure_by_chord_model.tables:
        warnings.append(
            "λ chord-figure > 0 but no fitted p(figure | chord) table is loaded; the term is inert — "
            "run `make fit-generator` on a chord-decoded build."
        )

    return warnings


def _with_warnings(message: str, warnings: list[str]) -> str:
    if not warnings:
        return message

    return message + "".join(f"\n⚠ {warning}" for warning in warnings)


def _chord_transition_model(
    request: SyntheticGenerationRequest,
    inputs: SyntheticInputs,
    *,
    chords: tuple[Chord, ...],
    scale_type: ScaleType,
) -> tuple[ChordTransitionModel, str | None]:
    if request.chord_model == "uniform":
        return uniform_transition_model(chords, self_transition_bias=request.self_transition_bias), None

    if request.chord_model == "empirical":
        empirical = inputs.fitted.chord_transition_model(scale_type)
        if empirical is not None:
            return empirical, None

        warning = (
            f"chord_model='empirical' requested but no fitted chord transitions for {scale_type.value}; "
            "using the functional prior instead — run `make fit-generator` on a chord-decoded build."
        )
        return _functional_transition_model(request, chords=chords, scale_type=scale_type), warning

    return _functional_transition_model(request, chords=chords, scale_type=scale_type), None


def _functional_transition_model(
    request: SyntheticGenerationRequest,
    *,
    chords: tuple[Chord, ...],
    scale_type: ScaleType,
) -> ChordTransitionModel:
    return functional_transition_model(
        chords,
        scale_type=scale_type,
        strength=request.functional_strength,
        self_transition_bias=request.self_transition_bias,
    )


def _hand_texture_config(request: SyntheticGenerationRequest) -> HandTextureConfig:
    return HandTextureConfig(
        right=HandTexture(request.right_texture),
        left=HandTexture(request.left_texture),
        accompaniment=AccompanimentConfig(
            rhythm=AccompanimentRhythm(request.accompaniment_rhythm),
            max_chord_notes=request.accompaniment_max_notes,
        ),
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


def _failure(
    duration_vocabulary: DurationVocabulary,
    message: str,
    *,
    warnings: list[str] | None = None,
) -> SyntheticGeneratedOutput:
    return SyntheticGeneratedOutput(
        segment=None,
        trace=GenerationTrace(samples=(), grid_count_per_bar=1, bar_count=0),
        duration_vocabulary=duration_vocabulary,
        decode_error=None,
        error=message,
        status_message=_with_warnings(message, warnings or []),
        status_kind="warn",
    )
