from fractions import Fraction

from musak_model.data.quantizer import quantize_duration, quantize_duration_to_id
from musak_model.tokens.duration import DurationVocabulary


def test_quantize_duration_to_id_keeps_exact_known_values(duration_vocabulary: DurationVocabulary) -> None:
    assert quantize_duration_to_id(
        Fraction(1, 4), vocabulary=duration_vocabulary
    ) == duration_vocabulary.fraction_to_id(Fraction(1, 4))
    assert quantize_duration_to_id(
        Fraction(3, 8), vocabulary=duration_vocabulary
    ) == duration_vocabulary.fraction_to_id(Fraction(3, 8))
    assert quantize_duration_to_id(
        Fraction(1, 12), vocabulary=duration_vocabulary
    ) == duration_vocabulary.fraction_to_id(Fraction(1, 12))


def test_quantize_duration_to_id_rounds_to_nearest_known_duration(duration_vocabulary: DurationVocabulary) -> None:
    duration_id = quantize_duration_to_id(Fraction(11, 32), vocabulary=duration_vocabulary)
    assert duration_vocabulary.id_to_fraction(duration_id) == Fraction(1, 3)

    duration_id = quantize_duration_to_id(Fraction(1, 9), vocabulary=duration_vocabulary)
    assert duration_vocabulary.id_to_fraction(duration_id) == Fraction(1, 8)


def test_quantize_duration_reports_quantization_error(duration_vocabulary: DurationVocabulary) -> None:
    result = quantize_duration(Fraction(1, 9), vocabulary=duration_vocabulary)

    assert result.original == Fraction(1, 9)
    assert result.quantized == Fraction(1, 8)
    assert result.error == Fraction(1, 72)
    assert result.exact is False
