from fractions import Fraction
from typing import Final

type RatioValue = Fraction | tuple[int, int]

RATIO_SEPARATOR_SYMBOL: Final[str] = "/"


def format_ratio(value: RatioValue, *, separator: str = RATIO_SEPARATOR_SYMBOL) -> str:
    if isinstance(value, Fraction):
        numerator, denominator = value.numerator, value.denominator
    elif isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"ratio value tuple must have exactly 2 elements, got {len(value)}")

        numerator, denominator = value
    else:
        raise TypeError(f"unsupported ratio value type: {type(value)}")

    return f"{numerator}{separator}{denominator}"
