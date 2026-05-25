from collections.abc import Callable

from musak_model.data.schema import Segment
from musak_model.tokens.schema import HoldToken, NoteToken, RestToken


def coalesce_optional_integer(value: int | None) -> int:
    return value if value is not None else 0


def collect_note_tokens(segment: Segment) -> list[NoteToken]:
    return [token for token in segment.tokens if isinstance(token, NoteToken)]


def calculate_note_fraction(notes: list[NoteToken], predicate: Callable[[NoteToken], bool]) -> float:
    if not notes:
        return 0.0

    return sum(predicate(note) for note in notes) / len(notes)


def calculate_token_fraction(
    segment: Segment,
    token_type: type[NoteToken] | type[RestToken] | type[HoldToken],
) -> float:
    if not segment.tokens:
        return 0.0

    return sum(isinstance(token, token_type) for token in segment.tokens) / len(segment.tokens)
