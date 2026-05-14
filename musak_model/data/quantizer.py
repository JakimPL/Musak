from dataclasses import dataclass
from fractions import Fraction

from musak_model.tokens.duration import DurationVocabulary


@dataclass(frozen=True)
class QuantizedDuration:
    duration_id: int
    original: Fraction
    quantized: Fraction

    @property
    def error(self) -> Fraction:
        return abs(self.quantized - self.original)

    @property
    def exact(self) -> bool:
        return self.error == 0


def quantize_duration(duration: Fraction, *, vocabulary: DurationVocabulary) -> QuantizedDuration:
    duration_id, quantized = vocabulary.find_closest(duration)
    return QuantizedDuration(duration_id=duration_id, original=duration, quantized=quantized)


def quantize_duration_to_id(duration: Fraction, *, vocabulary: DurationVocabulary) -> int:
    return quantize_duration(duration, vocabulary=vocabulary).duration_id
