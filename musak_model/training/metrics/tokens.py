from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor

from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    StartToken,
)
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.metrics.schema import BatchMetrics


class TokenKindId(IntEnum):
    NOTE = 0
    REST = 1
    HOLD = 2
    BAR = 3
    END = 4
    HAND = 5
    JOIN_WITH_PREVIOUS = 6
    START = 7


def batch_metrics_from_logits(
    logits: Tensor,
    *,
    target_token_ids: Tensor,
    token_padding_mask: Tensor,
    loss: Tensor,
    token_kind_ids: Tensor | None = None,
) -> BatchMetrics:
    valid_mask = ~token_padding_mask.reshape(-1)
    token_count = int(valid_mask.sum().item())
    if token_count == 0:
        raise ValueError("batch has no valid target tokens")

    predicted_token_ids = logits.argmax(dim=-1)
    exact_matches = (predicted_token_ids == target_token_ids) & ~token_padding_mask
    token_kind_match_count = _token_kind_match_count(
        predicted_token_ids,
        target_token_ids=target_token_ids,
        token_padding_mask=token_padding_mask,
        token_kind_ids=token_kind_ids,
    )
    return BatchMetrics(
        loss=float(loss.detach().item()),
        token_count=token_count,
        exact_match_count=int(exact_matches.sum().item()),
        token_kind_match_count=token_kind_match_count,
    )


def build_token_kind_ids(vocabulary: TokenVocabulary) -> Tensor:
    return torch.tensor(
        [_token_kind_id(vocabulary.id_to_token(token_id)) for token_id in range(vocabulary.vocabulary_size)],
        dtype=torch.long,
    )


def _token_kind_match_count(
    predicted_token_ids: Tensor,
    *,
    target_token_ids: Tensor,
    token_padding_mask: Tensor,
    token_kind_ids: Tensor | None,
) -> int | None:
    if token_kind_ids is None:
        return None

    device_token_kind_ids = token_kind_ids.to(predicted_token_ids.device)
    predicted_kind_ids = device_token_kind_ids[predicted_token_ids]
    target_kind_ids = device_token_kind_ids[target_token_ids]
    kind_matches = (predicted_kind_ids == target_kind_ids) & ~token_padding_mask
    return int(kind_matches.sum().item())


def _token_kind_id(token: object) -> int:
    match token:
        case NoteToken():
            return TokenKindId.NOTE
        case RestToken():
            return TokenKindId.REST
        case HoldToken():
            return TokenKindId.HOLD
        case BarToken():
            return TokenKindId.BAR
        case EndToken():
            return TokenKindId.END
        case HandToken():
            return TokenKindId.HAND
        case JoinWithPreviousToken():
            return TokenKindId.JOIN_WITH_PREVIOUS
        case StartToken():
            return TokenKindId.START
        case _:
            raise ValueError(f"unsupported token type: {type(token).__name__}")
