from collections import Counter
from collections.abc import Hashable
from typing import TypeVar

_DistributionKey = TypeVar("_DistributionKey", bound=Hashable)


def total_variation_distance(
    reference_counts: Counter[_DistributionKey],
    comparison_counts: Counter[_DistributionKey],
) -> float:
    reference_total = sum(reference_counts.values())
    if reference_total == 0:
        return 0.0

    comparison_total = sum(comparison_counts.values())
    keys = set(reference_counts) | set(comparison_counts)
    return 0.5 * sum(
        abs(
            (reference_counts[key] / reference_total)
            - (comparison_counts[key] / comparison_total if comparison_total > 0 else 0.0)
        )
        for key in keys
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)
