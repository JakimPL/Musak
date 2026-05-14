from typing import Final

from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.tokens.schema import ScaleType

SCALE_TYPE_TO_ID: Final[dict[ScaleType, int]] = {scale_type: index for index, scale_type in enumerate(ScaleType)}


def difficulty_level_to_id(difficulty_level: int | None) -> int | None:
    if difficulty_level is None:
        return None

    return difficulty_level


def scale_type_to_id(scale_type: ScaleType) -> int:
    try:
        return SCALE_TYPE_TO_ID[scale_type]
    except KeyError as exception:
        raise ValueError(f"unsupported scale type: {scale_type}") from exception


def time_signature_to_id(time_signature: tuple[int, int], *, vocabulary: TimeSignatureVocabulary) -> int:
    return vocabulary.time_signature_to_id(time_signature)
