from collections import Counter
from pathlib import Path

from musak_model.n_grams.profile.metrics.stats import mean, total_variation_distance
from musak_model.n_grams.profile.schema import FigureProfile
from musak_model.training.stages.figure_profiles.counts import iter_count_groups


def split_figure_profile_count_metrics(
    *,
    train_profile: FigureProfile,
    validation_profile: FigureProfile,
) -> dict[str, float]:
    return {
        "model/split/figure/count/train_samples": float(train_profile.metadata.sample_count),
        "model/split/figure/count/validation_samples": float(validation_profile.metadata.sample_count),
        "model/split/figure/count/train_profile_groups": float(len(train_profile.groups)),
        "model/split/figure/count/validation_profile_groups": float(len(validation_profile.groups)),
    }


def figure_distribution_metrics_from_csv(
    *,
    reference_path: Path,
    comparison_path: Path,
    metric_prefix: str,
) -> dict[str, float]:
    distances: list[float] = []
    comparison_groups = iter_count_groups(comparison_path)
    comparison_group = next(comparison_groups, None)
    for reference_group in iter_count_groups(reference_path):
        if not reference_group.counts:
            continue

        while comparison_group is not None and comparison_group.key < reference_group.key:
            comparison_group = next(comparison_groups, None)

        if comparison_group is not None and comparison_group.key == reference_group.key:
            comparison_counts = comparison_group.counts
            comparison_group = next(comparison_groups, None)
        else:
            comparison_counts = Counter()

        distances.append(total_variation_distance(reference_group.counts, comparison_counts))

    if not distances:
        return {f"{metric_prefix}/count/distribution_groups": 0.0}

    return {
        f"{metric_prefix}/count/distribution_groups": float(len(distances)),
        f"{metric_prefix}/mean/identity_total_variation_distance": mean(distances),
    }
