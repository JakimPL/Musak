from musak.config.defaults import (
    MAX_TIME_SIGNATURE_DENOMINATOR,
    MAX_TIME_SIGNATURE_NUMERATOR,
    MIN_TIME_SIGNATURE_DENOMINATOR,
    MIN_TIME_SIGNATURE_NUMERATOR,
)
from musak.modules.elements.misc import is_power_of_two

TimeSignatureType = tuple[int, int]


def validate_time_signature(time_signature: TimeSignatureType) -> TimeSignatureType:
    numerator, denominator = time_signature
    if not MIN_TIME_SIGNATURE_NUMERATOR <= numerator <= MAX_TIME_SIGNATURE_NUMERATOR:
        raise ValueError(
            f"time signature numerator must be between {MIN_TIME_SIGNATURE_NUMERATOR} and "
            f"{MAX_TIME_SIGNATURE_NUMERATOR}, got {numerator}"
        )

    if not is_power_of_two(denominator):
        raise ValueError(f"time signature denominator must be a power of two, got {denominator}")

    if not MIN_TIME_SIGNATURE_DENOMINATOR <= denominator <= MAX_TIME_SIGNATURE_DENOMINATOR:
        raise ValueError(
            f"time signature denominator must be between {MIN_TIME_SIGNATURE_DENOMINATOR} and "
            f"{MAX_TIME_SIGNATURE_DENOMINATOR}, got {denominator}"
        )

    return time_signature
