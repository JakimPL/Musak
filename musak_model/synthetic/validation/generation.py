from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from numpy.random import default_rng

from musak_model.data.schema import Segment
from musak_model.generation.constraints import GenerationConstraintError, GenerationConstraints
from musak_model.synthetic.fitting.form.fit import FormFittingConfig
from musak_model.synthetic.inputs import SyntheticInputs
from musak_model.synthetic.processes.density import RhythmicDensityConfig
from musak_model.synthetic.render.build import build_surface_renderer
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.motif import MotifConfig
from musak_model.synthetic.render.renderer import RenderedChord, SurfaceRenderer
from musak_model.synthetic.structure.form import FormPrior, FormSampler
from musak_model.synthetic.validation.config import SyntheticValidationConfig
from musak_model.tokens.schema import ScaleType

_LOGGER = logging.getLogger(__name__)
_SOURCE_FILE = Path("synthetic-validation")


@dataclass(frozen=True)
class GeneratedSample:
    scale_type: ScaleType
    seed: int
    segment: Segment | None
    chords: tuple[RenderedChord, ...]
    render_error: str | None


def generate_scale_samples(
    inputs: SyntheticInputs,
    config: SyntheticValidationConfig,
    *,
    scale_type: ScaleType,
    form_fitting: FormFittingConfig,
) -> list[GeneratedSample]:
    renderer = build_surface_renderer(
        inputs,
        render_config=_render_config(config),
        motif_config=MotifConfig.load().model_copy(update={"variation_budget": config.variation_budget}),
        density_config=RhythmicDensityConfig.load().model_copy(
            update={"amplitude": config.density_amplitude, "basis_count": config.density_basis_count}
        ),
    )
    prior = _select_prior(inputs, scale_type=scale_type, config=config, form_fitting=form_fitting)
    samples = [
        _render_sample(renderer, prior, config=config, scale_type=scale_type, seed=config.base_seed + index)
        for index in range(config.samples_per_scale)
    ]
    rendered = sum(sample.segment is not None for sample in samples)
    _LOGGER.info("Generated %s/%s %s samples", rendered, len(samples), scale_type.value)
    return samples


def _render_sample(
    renderer: SurfaceRenderer,
    prior: FormPrior,
    *,
    config: SyntheticValidationConfig,
    scale_type: ScaleType,
    seed: int,
) -> GeneratedSample:
    form = FormSampler(prior).sample(bar_count=config.bar_count, rng=default_rng(seed))
    constraints = GenerationConstraints(
        time_numerator=config.time_numerator,
        time_denominator=config.time_denominator,
        bar_count=config.bar_count,
    )
    try:
        result = renderer.render_plan(
            time_numerator=config.time_numerator,
            time_denominator=config.time_denominator,
            scale_root=config.scale_root,
            scale_type=scale_type,
            form=form,
            harmonic_slot_duration=Fraction(1, config.harmonic_slot_denominator),
            constraints=constraints,
            source_file=_SOURCE_FILE,
            rng=default_rng(seed),
        )
    except (GenerationConstraintError, ValueError) as exception:
        return GeneratedSample(scale_type=scale_type, seed=seed, segment=None, chords=(), render_error=str(exception))

    return GeneratedSample(
        scale_type=scale_type, seed=seed, segment=result.segment, chords=result.chords, render_error=None
    )


def _render_config(config: SyntheticValidationConfig) -> RenderConfig:
    return RenderConfig.load().model_copy(
        update={
            "commonness_bias": config.commonness_bias,
            "lambda_curve": config.lambda_curve,
            "lambda_harmonic": config.lambda_harmonic,
            "lambda_accent": config.lambda_accent,
            "lambda_similarity": config.lambda_similarity,
            "melodic_continuity": config.melodic_continuity,
        }
    )


def _select_prior(
    inputs: SyntheticInputs,
    *,
    scale_type: ScaleType,
    config: SyntheticValidationConfig,
    form_fitting: FormFittingConfig,
) -> FormPrior:
    if config.prior_source == "fitted":
        fitted = inputs.fitted.form_prior(scale_type)
        if fitted is not None:
            return fitted

    return form_fitting.fallback_prior
