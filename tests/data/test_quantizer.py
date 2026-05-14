from fractions import Fraction

from musak_model.data.quantizer import quantize_duration_to_id
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary


def _duration_vocabulary() -> DurationVocabulary:
    return DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))


def test_quantize_duration_to_id_keeps_exact_known_values() -> None:
    vocabulary = _duration_vocabulary()

    assert quantize_duration_to_id(Fraction(1, 4), vocabulary=vocabulary) == vocabulary.fraction_to_id(Fraction(1, 4))
    assert quantize_duration_to_id(Fraction(3, 8), vocabulary=vocabulary) == vocabulary.fraction_to_id(Fraction(3, 8))
    assert quantize_duration_to_id(Fraction(1, 12), vocabulary=vocabulary) == vocabulary.fraction_to_id(Fraction(1, 12))


def test_quantize_duration_to_id_rounds_to_nearest_known_duration() -> None:
    vocabulary = _duration_vocabulary()

    duration_id = quantize_duration_to_id(Fraction(11, 32), vocabulary=vocabulary)
    assert vocabulary.id_to_fraction(duration_id) == Fraction(1, 3)

    duration_id = quantize_duration_to_id(Fraction(1, 9), vocabulary=vocabulary)
    assert vocabulary.id_to_fraction(duration_id) == Fraction(1, 8)
