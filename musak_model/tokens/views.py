from collections.abc import Sequence

from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    NoteToken,
    RestToken,
    StartToken,
    Token,
)


def tokens_for_hand(
    tokens: Sequence[Token],
    *,
    hand: Hand,
    include_structure: bool = True,
) -> list[Token]:
    active_hand = Hand.RIGHT
    selected_tokens: list[Token] = []

    for token in tokens:
        if isinstance(token, HandToken):
            active_hand = token.hand
            continue

        if include_structure and isinstance(token, (StartToken, BarToken, EndToken)):
            selected_tokens.append(token)
            continue

        if active_hand == hand and isinstance(token, (NoteToken, RestToken, HoldToken)):
            selected_tokens.append(token)

    return selected_tokens
