from collections.abc import Iterable
from typing import TypeVar, cast

from tqdm.auto import tqdm

_T = TypeVar("_T")


def progress(
    values: Iterable[_T],
    *,
    description: str,
    unit: str,
    enabled: bool,
    total: int | None = None,
) -> Iterable[_T]:
    if not enabled:
        return values

    return cast(
        Iterable[_T],
        tqdm(
            values,
            total=total,
            desc=description,
            unit=unit,
        ),
    )
