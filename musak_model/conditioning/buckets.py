from __future__ import annotations

from fractions import Fraction
from typing import Final

DEFAULT_UNKNOWN_BUCKET_ID: Final[int] = 0


def threshold_bucket_vocabulary_size(thresholds: tuple[object, ...]) -> int:
    return len(thresholds) + 2


def optional_integer_threshold_bucket_id(
    value: int | None,
    thresholds: tuple[int, ...],
    *,
    unknown_id: int = DEFAULT_UNKNOWN_BUCKET_ID,
) -> int:
    if value is None:
        return unknown_id

    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return index

    return len(thresholds) + 1


def optional_float_threshold_bucket_id(
    value: float | None,
    thresholds: tuple[float, ...],
    *,
    unknown_id: int = DEFAULT_UNKNOWN_BUCKET_ID,
) -> int:
    if value is None:
        return unknown_id

    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return index

    return len(thresholds) + 1


def optional_fraction_threshold_bucket_id(
    value: Fraction | None,
    thresholds: tuple[Fraction, ...],
    *,
    unknown_id: int = DEFAULT_UNKNOWN_BUCKET_ID,
) -> int:
    if value is None:
        return unknown_id

    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return index

    return len(thresholds) + 1
