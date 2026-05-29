from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Final

from numpy.random import default_rng

from musak_model.data.schema import Segment
from musak_model.generation.constraints import GenerationConstraints
from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.profile.metrics.distribution import figure_distribution_metrics
from musak_model.synthetic.calibration.config import CalibrationConfig
from musak_model.synthetic.calibration.counts import segment_figure_counts
from musak_model.synthetic.calibration.schema import SweepResult
from musak_model.synthetic.substitution import SegmentGenerator

_METRIC_PREFIX: Final = "calibration"
_DISTRIBUTION_GROUPS_KEY: Final = f"{_METRIC_PREFIX}/count/distribution_groups"
_MEAN_TV_KEY: Final = f"{_METRIC_PREFIX}/mean/identity_total_variation_distance"
_SOURCE_FILE: Final = Path("calibration")


def run_sweep(
    *,
    generator: SegmentGenerator,
    reference_counts: FigureNGramCountsByScale,
    config: CalibrationConfig,
) -> list[SweepResult]:
    restricted_reference: FigureNGramCountsByScale = (
        {config.scale_type: reference_counts[config.scale_type]} if config.scale_type in reference_counts else {}
    )
    constraints = GenerationConstraints(
        time_numerator=config.time_numerator,
        time_denominator=config.time_denominator,
        bar_count=config.bar_count,
    )
    results: list[SweepResult] = []
    for lambda_curve, lambda_harm, lambda_accent in product(
        config.lambda_curve, config.lambda_harm, config.lambda_accent
    ):
        cell_generator = replace(
            generator,
            substitution_config=generator.substitution_config.model_copy(
                update={
                    "lambda_curve": lambda_curve,
                    "lambda_harm": lambda_harm,
                    "lambda_accent": lambda_accent,
                }
            ),
        )
        generated_counts = segment_figure_counts(
            _generate_segments(cell_generator, config=config, constraints=constraints),
            min_n=config.min_n,
            max_n=config.max_n,
            duration_vocabulary=generator.duration_vocabulary,
        )
        metrics = figure_distribution_metrics(
            reference_counts=restricted_reference,
            comparison_counts=generated_counts,
            metric_prefix=_METRIC_PREFIX,
        )
        results.append(
            SweepResult(
                lambda_curve=lambda_curve,
                lambda_harm=lambda_harm,
                lambda_accent=lambda_accent,
                distribution_groups=int(metrics[_DISTRIBUTION_GROUPS_KEY]),
                mean_total_variation_distance=metrics.get(_MEAN_TV_KEY),
            )
        )

    return results


def _generate_segments(
    generator: SegmentGenerator,
    *,
    config: CalibrationConfig,
    constraints: GenerationConstraints,
) -> list[Segment]:
    return [
        generator.generate(
            bar_count=config.bar_count,
            time_numerator=config.time_numerator,
            time_denominator=config.time_denominator,
            scale_root=config.scale_root,
            scale_type=config.scale_type,
            constraints=constraints,
            rng=default_rng(config.seed + sample_index),
            source_file=_SOURCE_FILE,
        )
        for sample_index in range(config.samples_per_config)
    ]
