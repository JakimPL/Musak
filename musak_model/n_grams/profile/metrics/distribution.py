from collections import Counter

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.metrics.stats import mean, total_variation_distance


def figure_distribution_metrics(
    *,
    reference_counts: FigureNGramCountsByScale,
    comparison_counts: FigureNGramCountsByScale,
    metric_prefix: str,
) -> dict[str, float]:
    distances: list[float] = []
    for scale_type, reference_counts_by_hand in reference_counts.items():
        for hand, reference_counts_by_length in reference_counts_by_hand.items():
            for figure_length, reference_figure_counts in reference_counts_by_length.items():
                if not reference_figure_counts:
                    continue

                comparison_figure_counts: Counter[FigureNGram] = (
                    comparison_counts.get(scale_type, {}).get(hand, {}).get(figure_length, Counter())
                )
                distances.append(total_variation_distance(reference_figure_counts, comparison_figure_counts))

    if not distances:
        return {f"{metric_prefix}/count/distribution_groups": 0.0}

    return {
        f"{metric_prefix}/count/distribution_groups": float(len(distances)),
        f"{metric_prefix}/mean/identity_total_variation_distance": mean(distances),
    }
