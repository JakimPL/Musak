from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from musak_model.evaluation.generation.figure_metrics import figure_profile_metrics
from musak_model.evaluation.generation.musical_metrics import musical_profile_metrics
from musak_model.evaluation.generation.rhythm.metrics import rhythm_profile_metrics
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.evaluation.generation.suite_metrics import suite_metrics
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.config import RhythmAnalysisConfig
from musak_model.n_grams.profile.loading import FigureProfileArtifacts
from musak_model.synthetic.validation.adapter import generation_sample
from musak_model.synthetic.validation.config import SyntheticValidationConfig
from musak_model.synthetic.validation.generation import GeneratedSample
from musak_model.synthetic.validation.options import metric_options
from musak_model.synthetic.validation.synthetic_metrics import synthetic_metrics
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType

_OVERALL_SUITE_NAME = "overall"


def validation_metrics(
    samples_by_scale: Mapping[ScaleType, Sequence[GeneratedSample]],
    *,
    config: SyntheticValidationConfig,
    artifacts: FigureProfileArtifacts | None,
    chord_vocabulary: ChordVocabularyConfig,
    duration_vocabulary: DurationVocabulary,
    rhythm_config: RhythmAnalysisConfig,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    pooled_samples: list[GenerationSample] = []
    for scale_type, generated in samples_by_scale.items():
        options = metric_options(config, scale_type)
        rendered = [sample for sample in generated if sample.segment is not None]
        generation_samples = [
            generation_sample(segment, options=options, duration_vocabulary=duration_vocabulary)
            for sample in rendered
            if (segment := sample.segment) is not None
        ]
        metrics.update(suite_metrics(scale_type.value, generation_samples))
        metrics.update(
            _namespace(
                figure_profile_metrics(
                    artifacts, samples=generation_samples, config=options, duration_vocabulary=duration_vocabulary
                ),
                scale_type,
            )
        )
        metrics.update(
            _namespace(
                rhythm_profile_metrics(
                    artifacts,
                    samples=generation_samples,
                    config=options,
                    rhythm_config=rhythm_config,
                    duration_vocabulary=duration_vocabulary,
                ),
                scale_type,
            )
        )
        metrics.update(
            _namespace(
                musical_profile_metrics(
                    samples=generation_samples, config=options, duration_vocabulary=duration_vocabulary
                ),
                scale_type,
            )
        )
        metrics.update(
            _namespace(
                synthetic_metrics(
                    rendered,
                    options=options,
                    chord_vocabulary=chord_vocabulary,
                    duration_vocabulary=duration_vocabulary,
                ),
                scale_type,
            )
        )
        metrics[f"generation/{scale_type.value}/count/attempted"] = float(len(generated))
        metrics[f"generation/{scale_type.value}/count/rendered"] = float(len(rendered))
        metrics[f"generation/{scale_type.value}/rate/render_error"] = _rate(
            sum(sample.render_error is not None for sample in generated), len(generated)
        )
        pooled_samples.extend(generation_samples)

    metrics.update(suite_metrics(_OVERALL_SUITE_NAME, pooled_samples))
    return {name: value for name, value in metrics.items() if math.isfinite(value)}


def _namespace(metrics: dict[str, float], scale_type: ScaleType) -> dict[str, float]:
    return {name.replace("generation/", f"generation/{scale_type.value}/", 1): value for name, value in metrics.items()}


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan

    return numerator / denominator
