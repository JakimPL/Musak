import math
from collections import Counter

from musak_model.n_grams.profile.metrics.stats import mean, total_variation_distance
from musak_model.n_grams.profile.rhythm.schema import RhythmCountCounter, RhythmGroupKey


def rhythm_reference_distribution_metrics(
    *,
    reference_counts: RhythmCountCounter,
    comparison_counts: RhythmCountCounter,
    metric_prefix: str,
) -> dict[str, float]:
    reference_groups = _group_counts(reference_counts)
    comparison_groups = _group_counts(comparison_counts)
    metrics: dict[str, float] = {}
    for kind in (
        "rhythm_ngram",
        "duration_value",
        "onset_grid_alignment",
        "duration_grid_alignment",
        "strong_beat_onset",
    ):
        kind_keys = sorted(key for key in reference_groups if key.kind == kind)
        distances = [
            total_variation_distance(reference_groups[key], comparison_groups.get(key, Counter()))
            for key in kind_keys
            if reference_groups[key]
        ]
        metrics[f"{metric_prefix}/count/{kind}_distribution_groups"] = float(len(distances))
        if distances:
            metrics[f"{metric_prefix}/mean/{kind}_total_variation_distance"] = mean(distances)

    duration_entropy_differences = [
        abs(_entropy(reference_group_counts) - _entropy(comparison_groups.get(key, Counter())))
        for key, reference_group_counts in reference_groups.items()
        if key.kind == "duration_value" and reference_group_counts
    ]
    if duration_entropy_differences:
        metrics[f"{metric_prefix}/mean/duration_entropy_absolute_error"] = mean(duration_entropy_differences)

    strong_beat_differences = [
        abs(
            _strong_beat_fraction(reference_group_counts) - _strong_beat_fraction(comparison_groups.get(key, Counter()))
        )
        for key, reference_group_counts in reference_groups.items()
        if key.kind == "strong_beat_onset" and reference_group_counts
    ]
    if strong_beat_differences:
        metrics[f"{metric_prefix}/mean/strong_beat_onset_fraction_absolute_error"] = mean(strong_beat_differences)

    return metrics


def _group_counts(counts: RhythmCountCounter) -> dict[RhythmGroupKey, Counter[str]]:
    groups: dict[RhythmGroupKey, Counter[str]] = {}
    for key, count in counts.items():
        group_key = RhythmGroupKey(
            scale_type=key.scale_type,
            time_signature=key.time_signature,
            hand=key.hand,
            kind=key.kind,
            parameter=key.parameter,
        )
        groups.setdefault(group_key, Counter())[key.value] += count

    return groups


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0

    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0)


def _strong_beat_fraction(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0

    return counts["strong"] / total
