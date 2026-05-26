from collections import Counter
from fractions import Fraction

import pytest

from musak_model.n_grams.figure.schema import FigureDegree, FigureNGram
from musak_model.n_grams.profile.metrics.reference.distribution import figure_reference_distribution_metrics
from musak_model.tokens.schema import Hand, ScaleType


def test_figure_reference_distribution_metrics_compare_common_rare_novel_and_shape_distributions() -> None:
    common_figure = _figure((((0, 0),), Fraction(1)), (((1, 0),), Fraction(1)))
    rare_figure = _figure((((0, 0),), Fraction(1)), (((2, 0),), Fraction(2)))
    novel_figure = _figure((((0, 0), (2, 0)), Fraction(1)), (((-1, 0),), Fraction(1)))
    reference_counts = {
        ScaleType.MAJOR: {
            Hand.RIGHT: {
                2: Counter({common_figure: 8, rare_figure: 2}),
            },
        },
    }
    comparison_counts = {
        ScaleType.MAJOR: {
            Hand.RIGHT: {
                2: Counter({common_figure: 3, rare_figure: 1, novel_figure: 1}),
            },
        },
    }

    metrics = figure_reference_distribution_metrics(
        reference_counts=reference_counts,
        comparison_counts=comparison_counts,
        metric_prefix="generation/figure",
        common_mass_threshold=0.8,
    )

    assert metrics == pytest.approx(
        {
            "generation/figure/count/distribution_groups": 1.0,
            "generation/figure/mean/identity_total_variation_distance": 0.2,
            "generation/figure/mean/common_figure_mass": 0.6,
            "generation/figure/mean/rare_figure_mass": 0.2,
            "generation/figure/mean/novel_figure_mass": 0.2,
            "generation/figure/mean/property_total_variation_distance": 0.2,
            "generation/figure/mean/contour_total_variation_distance": 0.2,
            "generation/figure/mean/duration_shape_total_variation_distance": 0.0,
        }
    )


def test_figure_reference_distribution_metrics_handles_empty_reference_counts() -> None:
    metrics = figure_reference_distribution_metrics(
        reference_counts={},
        comparison_counts={},
        metric_prefix="generation/figure",
        common_mass_threshold=0.8,
    )

    assert metrics == {"generation/figure/count/distribution_groups": 0.0}


def _figure(*onsets: tuple[tuple[FigureDegree, ...], Fraction]) -> FigureNGram:
    return FigureNGram(onsets=onsets)
