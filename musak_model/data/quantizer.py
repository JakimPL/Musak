from fractions import Fraction

from musak_model.tokens.duration import DurationVocabulary


def quantize_duration_to_id(duration: Fraction, *, vocabulary: DurationVocabulary) -> int:
    duration_id, _ = vocabulary.find_closest(duration)
    return duration_id
