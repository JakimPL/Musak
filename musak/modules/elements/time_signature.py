from typing import Final

from musak.modules.elements.misc import is_power_of_two

TimeSignatureType = tuple[int, int]

DEFAULT_TIME_SIGNATURE: Final[TimeSignatureType] = (4, 4)


def validate_time_signature(time_signature: TimeSignatureType) -> TimeSignatureType:
    numerator, denominator = time_signature
    if numerator <= 0:
        raise ValueError(f"time signature numerator must be positive, got {numerator}")

    if not is_power_of_two(denominator):
        raise ValueError(f"time signature denominator must be a power of two, got {denominator}")

    return time_signature
