from collections import Counter
from typing import Literal

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.schema import FigureProfile, FigureProfileGroup
from musak_model.tokens.schema import Hand, ScaleType

_PropertyName = Literal["monophonic", "chords_only", "in_scale"]
_PROPERTY_NAMES: tuple[_PropertyName, ...] = ("monophonic", "chords_only", "in_scale")


def figure_profile_comparison_metrics(
    *,
    reference_profile: FigureProfile,
    comparison_profile: FigureProfile,
    metric_prefix: str,
    comparison_sample_count_metric: str | None = None,
    require_comparison_samples: bool = False,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if comparison_sample_count_metric is not None:
        metrics[f"{metric_prefix}/count/{comparison_sample_count_metric}"] = float(
            comparison_profile.metadata.sample_count
        )

    reference_groups = {
        _figure_profile_group_key(group): group for group in reference_profile.groups if group.total > 0
    }
    comparison_groups = {_figure_profile_group_key(group): group for group in comparison_profile.groups}
    if not reference_groups or (require_comparison_samples and comparison_profile.metadata.sample_count == 0):
        return {**metrics, f"{metric_prefix}/count/comparable_groups": 0.0}

    total_relative_errors: list[float] = []
    property_rate_errors: dict[_PropertyName, list[float]] = {property_name: [] for property_name in _PROPERTY_NAMES}
    for key, reference_group in reference_groups.items():
        comparison_group = comparison_groups.get(key)
        comparison_total = comparison_group.total if comparison_group is not None else 0
        total_relative_errors.append(abs(comparison_total - reference_group.total) / reference_group.total)
        for property_name in _PROPERTY_NAMES:
            property_rate_errors[property_name].append(
                abs(_group_rate(comparison_group, property_name) - _group_rate(reference_group, property_name))
            )

    return {
        **metrics,
        f"{metric_prefix}/count/comparable_groups": float(len(reference_groups)),
        f"{metric_prefix}/mean/total_relative_abs_error": _mean(total_relative_errors),
        f"{metric_prefix}/mean/monophonic_rate_abs_error": _mean(property_rate_errors["monophonic"]),
        f"{metric_prefix}/mean/chords_only_rate_abs_error": _mean(property_rate_errors["chords_only"]),
        f"{metric_prefix}/mean/in_scale_rate_abs_error": _mean(property_rate_errors["in_scale"]),
    }


def figure_distribution_metrics(
    *,
    reference_counts: FigureNGramCountsByScale,
    comparison_counts: FigureNGramCountsByScale,
    metric_prefix: str,
) -> dict[str, float]:
    distances: list[float] = []
    for scale_type, reference_counts_by_hand in reference_counts.items():
        for hand, reference_counts_by_n in reference_counts_by_hand.items():
            for n, reference_figure_counts in reference_counts_by_n.items():
                if not reference_figure_counts:
                    continue

                comparison_figure_counts: Counter[FigureNGram] = (
                    comparison_counts.get(scale_type, {}).get(hand, {}).get(n, Counter())
                )
                distances.append(_total_variation_distance(reference_figure_counts, comparison_figure_counts))

    if not distances:
        return {f"{metric_prefix}/count/distribution_groups": 0.0}

    return {
        f"{metric_prefix}/count/distribution_groups": float(len(distances)),
        f"{metric_prefix}/mean/identity_total_variation_distance": _mean(distances),
    }


def _total_variation_distance(
    reference_counts: Counter[FigureNGram],
    comparison_counts: Counter[FigureNGram],
) -> float:
    reference_total = sum(reference_counts.values())
    if reference_total == 0:
        return 0.0

    comparison_total = sum(comparison_counts.values())
    figures = set(reference_counts) | set(comparison_counts)
    return 0.5 * sum(
        abs(
            (reference_counts[figure] / reference_total)
            - (comparison_counts[figure] / comparison_total if comparison_total > 0 else 0.0)
        )
        for figure in figures
    )


def _figure_profile_group_key(group: FigureProfileGroup) -> tuple[ScaleType, Hand, int]:
    return group.scale_type, group.hand, group.n


def _group_rate(
    group: FigureProfileGroup | None,
    field_name: _PropertyName,
) -> float:
    if group is None or group.total == 0:
        return 0.0

    match field_name:
        case "monophonic":
            return group.monophonic / group.total
        case "chords_only":
            return group.chords_only / group.total
        case "in_scale":
            return group.in_scale / group.total

    raise ValueError(f"unknown figure profile group field: {field_name}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
