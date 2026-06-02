from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from numpy.random import default_rng

from musak_model.data.schema import Segment
from musak_model.generation.constraints import GenerationConstraintError, GenerationConstraints
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.fitting.form.fit import FormFittingConfig
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.motif import MotifConfig
from musak_model.synthetic.render.renderer import SurfaceRenderer
from musak_model.synthetic.structure.form import FormPrior, FormSampler, FormTree
from musak_model.synthetic.structure.harmony_grammar import HarmonyGrammarConfig, HarmonyGrammarSampler
from musak_model.synthetic.structure.meter import MetricalGrammarConfig, MetricalTreeSampler
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType
from notebooks.utils.model_output import segment_decode_error
from notebooks.utils.synthetic import SyntheticInputs

_SOURCE_FILE = Path("synthetic-form")


@dataclass(frozen=True)
class FormRenderRequest:
    scale_root: int
    scale_type: str
    time_numerator: int
    time_denominator: int
    bar_count: int
    seed: int
    harmonic_slot_denominator: int
    prior_source: str
    commonness_bias: float
    lambda_curve: float
    lambda_harmonic: float
    lambda_accent: float
    lambda_similarity: float
    variation_budget: float


@dataclass(frozen=True)
class FormRenderOutput:
    segment: Segment | None
    form: FormTree | None
    duration_vocabulary: DurationVocabulary
    scale_root: int
    scale_type: ScaleType
    decode_error: str | None
    status_message: str
    status_kind: Literal["success", "warn"]


def render_form_segment(
    inputs: SyntheticInputs,
    request: FormRenderRequest,
    *,
    form_fitting: FormFittingConfig,
) -> FormRenderOutput:
    scale_type = ScaleType(request.scale_type)
    prior, warning = _select_prior(
        inputs, scale_type=scale_type, prior_source=request.prior_source, form_fitting=form_fitting
    )
    render_config = RenderConfig.load().model_copy(
        update={
            "commonness_bias": request.commonness_bias,
            "lambda_curve": request.lambda_curve,
            "lambda_harmonic": request.lambda_harmonic,
            "lambda_accent": request.lambda_accent,
            "lambda_similarity": request.lambda_similarity,
        }
    )
    motif_config = MotifConfig.load().model_copy(update={"variation_budget": request.variation_budget})
    renderer = _build_renderer(inputs, render_config=render_config, motif_config=motif_config)
    form = FormSampler(prior).sample(bar_count=request.bar_count, rng=default_rng(request.seed))
    constraints = GenerationConstraints(
        time_numerator=request.time_numerator,
        time_denominator=request.time_denominator,
        bar_count=request.bar_count,
    )
    try:
        segment = renderer.render(
            time_numerator=request.time_numerator,
            time_denominator=request.time_denominator,
            scale_root=request.scale_root,
            scale_type=scale_type,
            form=form,
            harmonic_slot_duration=Fraction(1, request.harmonic_slot_denominator),
            constraints=constraints,
            source_file=_SOURCE_FILE,
            rng=default_rng(request.seed),
        )
    except (GenerationConstraintError, ValueError) as exception:
        return FormRenderOutput(
            segment=None,
            form=form,
            duration_vocabulary=inputs.duration_vocabulary,
            scale_root=request.scale_root,
            scale_type=scale_type,
            decode_error=None,
            status_message=_with_warning(f"Render failed: {exception}", warning),
            status_kind="warn",
        )

    decode_error = segment_decode_error(segment, duration_vocabulary=inputs.duration_vocabulary)
    base_status = (
        f"Rendered {segment.bar_count} bar(s), {len(segment.tokens)} tokens, {len(form.phrases)} phrase(s) | "
        f"decode error: {decode_error or '-'}"
    )
    return FormRenderOutput(
        segment=segment,
        form=form,
        duration_vocabulary=inputs.duration_vocabulary,
        scale_root=request.scale_root,
        scale_type=scale_type,
        decode_error=decode_error,
        status_message=_with_warning(base_status, warning),
        status_kind="success" if decode_error is None and warning is None else "warn",
    )


def _build_renderer(
    inputs: SyntheticInputs,
    *,
    render_config: RenderConfig,
    motif_config: MotifConfig,
) -> SurfaceRenderer:
    chord_vocabulary = ChordVocabularyConfig.load()
    return SurfaceRenderer(
        config=render_config,
        metrical_sampler=MetricalTreeSampler(config=MetricalGrammarConfig.load()),
        harmony_sampler=HarmonyGrammarSampler(config=HarmonyGrammarConfig.load(), vocabulary=chord_vocabulary),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig.load(), overrides=inputs.fitted.register_overrides
        ),
        figure_vocabulary=inputs.figure_vocabulary,
        duration_vocabulary=inputs.duration_vocabulary,
        chord_vocabulary=chord_vocabulary,
        motif_config=motif_config,
    )


def _select_prior(
    inputs: SyntheticInputs,
    *,
    scale_type: ScaleType,
    prior_source: str,
    form_fitting: FormFittingConfig,
) -> tuple[FormPrior, str | None]:
    if prior_source == "fitted":
        fitted = inputs.fitted.form_prior(scale_type)
        if fitted is not None:
            return fitted, None

        return form_fitting.fallback_prior, (
            f"No fitted form prior for {scale_type.value}; using the fallback prior — run "
            "`extract_form_statistics` then `make fit-generator`."
        )

    return form_fitting.fallback_prior, None


def _with_warning(message: str, warning: str | None) -> str:
    return message if warning is None else f"{message}\n⚠ {warning}"
