from musak_shared.misc import is_power_of_two

TimeSignatureType = tuple[int, int]


def validate_time_positive(numerator: int, denominator: int) -> None:
    if numerator <= 0:
        raise ValueError(f"time signature numerator and denominator must be positive, got ({numerator}, {denominator})")

    if denominator <= 0:
        raise ValueError(f"time signature numerator and denominator must be positive, got ({numerator}, {denominator})")


def validate_time_denominator(denominator: int) -> None:
    if not is_power_of_two(denominator):
        raise ValueError("time signature denominator must be a power of two")


def validate_time_signature(time_signature: TimeSignatureType) -> None:
    numerator, denominator = time_signature
    validate_time_positive(numerator, denominator)
    validate_time_denominator(denominator)
