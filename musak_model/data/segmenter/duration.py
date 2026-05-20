from fractions import Fraction

from musak_model.data.quantizer import quantize_duration
from musak_model.data.schema import SegmentIneligibilityReason
from musak_model.data.segmenter.errors import TokenizationIneligibilityError
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand


def exact_duration_id(duration: Fraction, *, vocabulary: DurationVocabulary) -> int:
    quantized = quantize_duration(duration, vocabulary=vocabulary)
    if not quantized.exact:
        raise TokenizationIneligibilityError(
            f"unsupported duration {duration}: closest supported duration is {quantized.quantized}",
            reason=SegmentIneligibilityReason.QUANTIZATION_ERROR,
        )

    return quantized.duration_id


def exact_duration(
    duration: Fraction,
    *,
    vocabulary: DurationVocabulary,
    bar_index: int,
    hand: Hand,
    context: str,
) -> Fraction:
    quantized = quantize_duration(duration, vocabulary=vocabulary)
    if not quantized.exact:
        raise TokenizationIneligibilityError(
            (
                f"unsupported {context} duration in {hand.value} hand at bar {bar_index}: "
                f"{duration} would quantize to {quantized.quantized}"
            ),
            reason=SegmentIneligibilityReason.QUANTIZATION_ERROR,
        )

    return quantized.quantized
