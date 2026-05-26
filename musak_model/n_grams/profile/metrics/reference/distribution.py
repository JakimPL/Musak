from collections import Counter
from collections.abc import Callable, Hashable

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureDegree, FigureNGram
from musak_model.n_grams.profile.metrics.reference.frequency_mass import FigureFrequencyMass
from musak_model.n_grams.profile.metrics.reference.group_metrics import FigureReferenceGroupMetrics
from musak_model.n_grams.profile.metrics.stats import mean, total_variation_distance

type _FigureShape = tuple[Hashable, ...]


def figure_reference_distribution_metrics(
    *,
    reference_counts: FigureNGramCountsByScale,
    comparison_counts: FigureNGramCountsByScale,
    metric_prefix: str,
    common_mass_threshold: float,
) -> dict[str, float]:
    group_metrics = _figure_reference_group_metrics(
        reference_counts=reference_counts,
        comparison_counts=comparison_counts,
        common_mass_threshold=common_mass_threshold,
    )
    if not group_metrics:
        return {f"{metric_prefix}/count/distribution_groups": 0.0}

    return {
        f"{metric_prefix}/count/distribution_groups": float(len(group_metrics)),
        f"{metric_prefix}/mean/identity_total_variation_distance": mean(
            [group.identity_total_variation_distance for group in group_metrics]
        ),
        f"{metric_prefix}/mean/common_figure_mass": mean([group.common_figure_mass for group in group_metrics]),
        f"{metric_prefix}/mean/rare_figure_mass": mean([group.rare_figure_mass for group in group_metrics]),
        f"{metric_prefix}/mean/novel_figure_mass": mean([group.novel_figure_mass for group in group_metrics]),
        f"{metric_prefix}/mean/property_total_variation_distance": mean(
            [group.property_total_variation_distance for group in group_metrics]
        ),
        f"{metric_prefix}/mean/contour_total_variation_distance": mean(
            [group.contour_total_variation_distance for group in group_metrics]
        ),
        f"{metric_prefix}/mean/duration_shape_total_variation_distance": mean(
            [group.duration_shape_total_variation_distance for group in group_metrics]
        ),
    }


def _figure_reference_group_metrics(
    *,
    reference_counts: FigureNGramCountsByScale,
    comparison_counts: FigureNGramCountsByScale,
    common_mass_threshold: float,
) -> list[FigureReferenceGroupMetrics]:
    group_metrics: list[FigureReferenceGroupMetrics] = []
    for scale_type, reference_counts_by_hand in reference_counts.items():
        for hand, reference_counts_by_length in reference_counts_by_hand.items():
            for figure_length, reference_figure_counts in reference_counts_by_length.items():
                if not reference_figure_counts:
                    continue

                comparison_figure_counts: Counter[FigureNGram] = (
                    comparison_counts.get(scale_type, {}).get(hand, {}).get(figure_length, Counter())
                )
                mass = _figure_frequency_mass(
                    reference_figure_counts,
                    comparison_figure_counts,
                    common_mass_threshold=common_mass_threshold,
                )
                group_metrics.append(
                    FigureReferenceGroupMetrics(
                        identity_total_variation_distance=total_variation_distance(
                            reference_figure_counts,
                            comparison_figure_counts,
                        ),
                        common_figure_mass=mass.common,
                        rare_figure_mass=mass.rare,
                        novel_figure_mass=mass.novel,
                        property_total_variation_distance=_transformed_total_variation_distance(
                            reference_figure_counts,
                            comparison_figure_counts,
                            transform=_property_shape,
                        ),
                        contour_total_variation_distance=_transformed_total_variation_distance(
                            reference_figure_counts,
                            comparison_figure_counts,
                            transform=_contour_shape,
                        ),
                        duration_shape_total_variation_distance=_transformed_total_variation_distance(
                            reference_figure_counts,
                            comparison_figure_counts,
                            transform=_duration_shape,
                        ),
                    )
                )

    return group_metrics


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
    reference_counts: Counter[FigureNGram],
    comparison_counts: Counter[FigureNGram],
    *,
    common_mass_threshold: float,
) -> FigureFrequencyMass:
    comparison_total = sum(comparison_counts.values())
    if comparison_total == 0:
        return FigureFrequencyMass(common=0.0, rare=0.0, novel=0.0)

    common_figures = _common_figures(reference_counts, common_mass_threshold=common_mass_threshold)
    reference_figures = set(reference_counts)
    common_count = sum(count for figure, count in comparison_counts.items() if figure in common_figures)
    known_rare_count = sum(
        count
        for figure, count in comparison_counts.items()
        if figure in reference_figures and figure not in common_figures
    )
    novel_count = sum(count for figure, count in comparison_counts.items() if figure not in reference_figures)
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
