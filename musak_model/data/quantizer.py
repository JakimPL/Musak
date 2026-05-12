from fractions import Fraction

from musak_model.tokens.schema import DURATION_FRACTIONS, DurationClass


def quantize_duration(duration: Fraction) -> DurationClass:
    return min(
        DURATION_FRACTIONS,
        key=lambda duration_class: abs(DURATION_FRACTIONS[duration_class] - duration),
    )
