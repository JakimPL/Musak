from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureDegree, FigureNGram
from musak_model.n_grams.profile.metrics.reference.frequency_mass import FigureFrequencyMass
from musak_model.n_grams.profile.metrics.reference.group_metrics import FigureReferenceGroupMetrics
from musak_model.n_grams.profile.metrics.stats import mean, total_variation_distance
from musak_model.tokens.schema import Hand, ScaleType

type _FigureShape = tuple[Hashable, ...]
_DistributionKey = TypeVar("_DistributionKey", bound=Hashable)


@dataclass(frozen=True)
class FigureReferenceGroupKey:
    scale_type: ScaleType
    hand: Hand
    n: int


@dataclass(frozen=True)
class FigureReferenceDistributionGroup:
    key: FigureReferenceGroupKey
    reference_total: int
    figure_probabilities: Mapping[FigureNGram, float]
    property_probabilities: Mapping[_FigureShape, float]
    contour_probabilities: Mapping[_FigureShape, float]
    duration_shape_probabilities: Mapping[_FigureShape, float]
    reference_figures: frozenset[FigureNGram]
    common_figures: frozenset[FigureNGram]


@dataclass(frozen=True)
class FigureReferenceDistributionIndex:
    groups: Mapping[FigureReferenceGroupKey, FigureReferenceDistributionGroup]


def figure_reference_distribution_metrics(
    *,
    reference_counts: FigureNGramCountsByScale,
    comparison_counts: FigureNGramCountsByScale,
    metric_prefix: str,
    common_mass_threshold: float,
) -> dict[str, float]:
    index = build_figure_reference_distribution_index(
        reference_counts=reference_counts,
        common_mass_threshold=common_mass_threshold,
    )
    weighted_metrics = [
        (
            1,
            _figure_reference_group_metrics(
                reference_group=reference_group,
                comparison_counts=comparison_counts.get(reference_group.key.scale_type, {})
                .get(reference_group.key.hand, {})
                .get(reference_group.key.n, Counter()),
            ),
        )
        for reference_group in index.groups.values()
    ]
    return _aggregate_reference_group_metrics(metric_prefix=metric_prefix, weighted_metrics=weighted_metrics)


def figure_reference_alignment_metrics(
    *,
    reference_counts: FigureNGramCountsByScale,
    comparison_counts: FigureNGramCountsByScale,
    metric_prefix: str,
    common_mass_threshold: float,
) -> dict[str, float]:
    index = build_figure_reference_distribution_index(
        reference_counts=reference_counts,
        common_mass_threshold=common_mass_threshold,
    )
    weighted_metrics: list[tuple[int, FigureReferenceGroupMetrics]] = []
    for key, comparison_figure_counts in _iter_comparison_groups(comparison_counts):
        if not comparison_figure_counts:
            continue

        reference_group = index.groups.get(key)
        if reference_group is None:
            continue

        weighted_metrics.append(
            (
                sum(comparison_figure_counts.values()),
                _figure_reference_group_metrics(
                    reference_group=reference_group,
                    comparison_counts=comparison_figure_counts,
                ),
            )
        )

    return _aggregate_reference_group_metrics(metric_prefix=metric_prefix, weighted_metrics=weighted_metrics)


def build_figure_reference_distribution_index(
    *,
    reference_counts: FigureNGramCountsByScale,
    common_mass_threshold: float,
) -> FigureReferenceDistributionIndex:
    groups: dict[FigureReferenceGroupKey, FigureReferenceDistributionGroup] = {}
    for key, reference_figure_counts in _iter_comparison_groups(reference_counts):
        reference_total = sum(reference_figure_counts.values())
        if reference_total == 0:
            continue

        groups[key] = FigureReferenceDistributionGroup(
            key=key,
            reference_total=reference_total,
            figure_probabilities=_probability_distribution(reference_figure_counts),
            property_probabilities=_transformed_probability_distribution(
                reference_figure_counts,
                transform=_property_shape,
            ),
            contour_probabilities=_transformed_probability_distribution(
                reference_figure_counts,
                transform=_contour_shape,
            ),
            duration_shape_probabilities=_transformed_probability_distribution(
                reference_figure_counts,
                transform=_duration_shape,
            ),
            reference_figures=frozenset(reference_figure_counts),
            common_figures=frozenset(
                _common_figures(reference_figure_counts, common_mass_threshold=common_mass_threshold)
            ),
        )

    return FigureReferenceDistributionIndex(groups=groups)


def _figure_reference_group_metrics(
    *,
    reference_group: FigureReferenceDistributionGroup,
    comparison_counts: Counter[FigureNGram],
) -> FigureReferenceGroupMetrics:
    mass = _figure_frequency_mass(reference_group, comparison_counts)
    return FigureReferenceGroupMetrics(
        identity_total_variation_distance=_total_variation_distance_from_reference_probabilities(
            reference_group.figure_probabilities,
            comparison_counts,
        ),
        common_figure_mass=mass.common,
        rare_figure_mass=mass.rare,
        novel_figure_mass=mass.novel,
        property_total_variation_distance=_transformed_total_variation_distance_from_reference_probabilities(
            reference_group.property_probabilities,
            comparison_counts,
            transform=_property_shape,
        ),
        contour_total_variation_distance=_transformed_total_variation_distance_from_reference_probabilities(
            reference_group.contour_probabilities,
            comparison_counts,
            transform=_contour_shape,
        ),
        duration_shape_total_variation_distance=_transformed_total_variation_distance_from_reference_probabilities(
            reference_group.duration_shape_probabilities,
            comparison_counts,
            transform=_duration_shape,
        ),
    )


def _aggregate_reference_group_metrics(
    *,
    metric_prefix: str,
    weighted_metrics: list[tuple[int, FigureReferenceGroupMetrics]],
) -> dict[str, float]:
    if not weighted_metrics:
        return {f"{metric_prefix}/count/distribution_groups": 0.0}

    return {
        f"{metric_prefix}/count/distribution_groups": float(len(weighted_metrics)),
        f"{metric_prefix}/mean/identity_total_variation_distance": _weighted_mean(
            weighted_metrics,
            lambda group: group.identity_total_variation_distance,
        ),
        f"{metric_prefix}/mean/common_figure_mass": _weighted_mean(
            weighted_metrics,
            lambda group: group.common_figure_mass,
        ),
        f"{metric_prefix}/mean/rare_figure_mass": _weighted_mean(
            weighted_metrics,
            lambda group: group.rare_figure_mass,
        ),
        f"{metric_prefix}/mean/novel_figure_mass": _weighted_mean(
            weighted_metrics,
            lambda group: group.novel_figure_mass,
        ),
        f"{metric_prefix}/mean/property_total_variation_distance": _weighted_mean(
            weighted_metrics,
            lambda group: group.property_total_variation_distance,
        ),
        f"{metric_prefix}/mean/contour_total_variation_distance": _weighted_mean(
            weighted_metrics,
            lambda group: group.contour_total_variation_distance,
        ),
        f"{metric_prefix}/mean/duration_shape_total_variation_distance": _weighted_mean(
            weighted_metrics,
            lambda group: group.duration_shape_total_variation_distance,
        ),
    }


def _transformed_total_variation_distance(
    reference_counts: Counter[FigureNGram],
    comparison_counts: Counter[FigureNGram],
    *,
    transform: Callable[[FigureNGram], _FigureShape],
) -> float:
    return total_variation_distance(
        _transform_counts(reference_counts, transform=transform),
        _transform_counts(comparison_counts, transform=transform),
    )


def _transformed_total_variation_distance_from_reference_probabilities(
    reference_probabilities: Mapping[_FigureShape, float],
    comparison_counts: Counter[FigureNGram],
    *,
    transform: Callable[[FigureNGram], _FigureShape],
) -> float:
    return _total_variation_distance_from_reference_probabilities(
        reference_probabilities,
        _transform_counts(comparison_counts, transform=transform),
    )


def _transform_counts(
    counts: Counter[FigureNGram],
    *,
    transform: Callable[[FigureNGram], _FigureShape],
) -> Counter[_FigureShape]:
    transformed: Counter[_FigureShape] = Counter()
    for figure, count in counts.items():
        transformed[transform(figure)] += count

    return transformed


def _figure_frequency_mass(
    reference_group: FigureReferenceDistributionGroup,
    comparison_counts: Counter[FigureNGram],
) -> FigureFrequencyMass:
    comparison_total = sum(comparison_counts.values())
    if comparison_total == 0:
        return FigureFrequencyMass(common=0.0, rare=0.0, novel=0.0)

    common_count = sum(count for figure, count in comparison_counts.items() if figure in reference_group.common_figures)
    known_rare_count = sum(
        count
        for figure, count in comparison_counts.items()
        if figure in reference_group.reference_figures and figure not in reference_group.common_figures
    )
    novel_count = sum(
        count for figure, count in comparison_counts.items() if figure not in reference_group.reference_figures
    )
    return FigureFrequencyMass(
        common=common_count / comparison_total,
        rare=known_rare_count / comparison_total,
        novel=novel_count / comparison_total,
    )


def _common_figures(
    reference_counts: Counter[FigureNGram],
    *,
    common_mass_threshold: float,
) -> set[FigureNGram]:
    reference_total = sum(reference_counts.values())
    if reference_total == 0:
        return set()

    selected: set[FigureNGram] = set()
    selected_total = 0
    for figure, count in reference_counts.most_common():
        selected.add(figure)
        selected_total += count
        if selected_total / reference_total >= common_mass_threshold:
            break

    return selected


def _probability_distribution(
    counts: Counter[_DistributionKey],
) -> dict[_DistributionKey, float]:
    total = sum(counts.values())
    if total == 0:
        return {}

    return {key: count / total for key, count in counts.items()}


def _transformed_probability_distribution(
    counts: Counter[FigureNGram],
    *,
    transform: Callable[[FigureNGram], _FigureShape],
) -> dict[_FigureShape, float]:
    return _probability_distribution(_transform_counts(counts, transform=transform))


def _total_variation_distance_from_reference_probabilities(
    reference_probabilities: Mapping[_DistributionKey, float],
    comparison_counts: Counter[_DistributionKey],
) -> float:
    comparison_total = sum(comparison_counts.values())
    if not reference_probabilities:
        return 0.0

    if comparison_total == 0:
        return 0.5

    overlap = sum(
        min(reference_probabilities.get(key, 0.0), count / comparison_total) for key, count in comparison_counts.items()
    )
    return 1.0 - overlap


def _iter_comparison_groups(
    counts_by_scale: FigureNGramCountsByScale,
) -> tuple[tuple[FigureReferenceGroupKey, Counter[FigureNGram]], ...]:
    return tuple(
        (
            FigureReferenceGroupKey(scale_type=scale_type, hand=hand, n=figure_length),
            figure_counts,
        )
        for scale_type, counts_by_hand in counts_by_scale.items()
        for hand, counts_by_length in counts_by_hand.items()
        for figure_length, figure_counts in counts_by_length.items()
    )


def _weighted_mean(
    weighted_metrics: list[tuple[int, FigureReferenceGroupMetrics]],
    selector: Callable[[FigureReferenceGroupMetrics], float],
) -> float:
    total_weight = sum(weight for weight, _ in weighted_metrics)
    if total_weight == 0:
        return mean([selector(group_metrics) for _, group_metrics in weighted_metrics])

    return sum(weight * selector(group_metrics) for weight, group_metrics in weighted_metrics) / total_weight


def _property_shape(figure: FigureNGram) -> _FigureShape:
    return (figure.monophonic, figure.chords_only, figure.in_scale)


def _contour_shape(figure: FigureNGram) -> _FigureShape:
    representative_degrees = tuple(_representative_degree(degrees) for degrees, _ in figure.onsets)
    return tuple(
        (next_degree > current_degree) - (next_degree < current_degree)
        for current_degree, next_degree in zip(representative_degrees, representative_degrees[1:])
    )


def _duration_shape(figure: FigureNGram) -> _FigureShape:
    return tuple(duration for _, duration in figure.onsets)


def _representative_degree(degrees: tuple[FigureDegree, ...]) -> int:
    return min(relative_position for relative_position, _ in degrees)
