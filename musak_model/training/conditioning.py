from typing import Final

from musak_model.tokens.schema import VALID_TIME_SIGNATURES, ScaleType

SCALE_TYPE_TO_ID: Final[dict[ScaleType, int]] = {scale_type: index for index, scale_type in enumerate(ScaleType)}
TIME_SIGNATURE_TO_ID: Final[dict[tuple[int, int], int]] = {
    time_signature: index for index, time_signature in enumerate(VALID_TIME_SIGNATURES)
}


def difficulty_level_to_id(difficulty_level: int | None) -> int | None:
    if difficulty_level is None:
        return None

    return difficulty_level - 1


def scale_type_to_id(scale_type: ScaleType) -> int:
    try:
        return SCALE_TYPE_TO_ID[scale_type]
    except KeyError as exception:
        raise ValueError(f"unsupported scale type: {scale_type}") from exception


def time_signature_to_id(time_signature: tuple[int, int]) -> int:
    try:
        return TIME_SIGNATURE_TO_ID[time_signature]
    except KeyError as exception:
        raise ValueError(f"unsupported time signature: {time_signature}") from exception
