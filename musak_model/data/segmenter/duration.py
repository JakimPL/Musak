from fractions import Fraction
from typing import NamedTuple

from musak_model.data.quantizer import quantize_duration
from musak_model.data.schema import SegmentIneligibilityReason
from musak_model.data.segmenter.errors import TokenizationIneligibilityError
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand


class ResolvedDuration(NamedTuple):
    original: Fraction
    duration_id: int
    quantized: Fraction


def resolve_exact_duration(
    duration: Fraction,
    *,
    vocabulary: DurationVocabulary,
    bar_index: int,
    hand: Hand,
    context: str,
) -> ResolvedDuration:
    duration_id = vocabulary.duration_id_or_none(duration)
    if duration_id is not None:
        return ResolvedDuration(original=duration, duration_id=duration_id, quantized=duration)

    quantized = quantize_duration(duration, vocabulary=vocabulary)
    raise TokenizationIneligibilityError(
        (
            f"unsupported {context} duration in {hand.value} hand at bar {bar_index}: "
            f"{duration} would quantize to {quantized.quantized}"
        ),
        reason=SegmentIneligibilityReason.QUANTIZATION_ERROR,
    )


def exact_duration_id(duration: Fraction, *, vocabulary: DurationVocabulary) -> int:
    duration_id = vocabulary.duration_id_or_none(duration)
    if duration_id is not None:
        return duration_id

    quantized = quantize_duration(duration, vocabulary=vocabulary)
    raise TokenizationIneligibilityError(
        f"unsupported duration {duration}: closest supported duration is {quantized.quantized}",
        reason=SegmentIneligibilityReason.QUANTIZATION_ERROR,
    )


def exact_duration(
    duration: Fraction,
    *,
    vocabulary: DurationVocabulary,
    bar_index: int,
    hand: Hand,
    context: str,
) -> Fraction:
    return resolve_exact_duration(
        duration,
        vocabulary=vocabulary,
        bar_index=bar_index,
        hand=hand,
        context=context,
    ).quantized
