from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from musak_model.tokens.factorized import ABSENT_ATTRIBUTE_ID, TokenAttributes, token_ids_to_attributes
from musak_model.tokens.vocabulary import TokenVocabulary


@dataclass(frozen=True)
class TokenAttributeTargetTensors:
    kind_ids: Tensor
    degree_ids: Tensor
    accidental_ids: Tensor
    octave_offset_ids: Tensor
    duration_ids: Tensor
    hand_ids: Tensor

    def to(self, device: torch.device) -> TokenAttributeTargetTensors:
        return TokenAttributeTargetTensors(
            kind_ids=self.kind_ids.to(device),
            degree_ids=self.degree_ids.to(device),
            accidental_ids=self.accidental_ids.to(device),
            octave_offset_ids=self.octave_offset_ids.to(device),
            duration_ids=self.duration_ids.to(device),
            hand_ids=self.hand_ids.to(device),
        )

    @property
    def shape(self) -> torch.Size:
        return self.kind_ids.shape


def token_attribute_targets_from_token_ids(
    token_ids: Tensor,
    *,
    vocabulary: TokenVocabulary,
) -> TokenAttributeTargetTensors:
    token_id_values = [int(token_id.item()) for token_id in token_ids]
    return token_attribute_targets_from_attributes(token_ids_to_attributes(token_id_values, vocabulary=vocabulary))


def token_attribute_targets_from_attributes(attributes: list[TokenAttributes]) -> TokenAttributeTargetTensors:
    return TokenAttributeTargetTensors(
        kind_ids=_tensor_from_attribute_values([attribute.kind_id for attribute in attributes]),
        degree_ids=_tensor_from_attribute_values([attribute.degree_id for attribute in attributes]),
        accidental_ids=_tensor_from_attribute_values([attribute.accidental_id for attribute in attributes]),
        octave_offset_ids=_tensor_from_attribute_values([attribute.octave_offset_id for attribute in attributes]),
        duration_ids=_tensor_from_attribute_values([attribute.duration_id for attribute in attributes]),
        hand_ids=_tensor_from_attribute_values([attribute.hand_id for attribute in attributes]),
    )


def pad_token_attribute_targets(
    targets: list[TokenAttributeTargetTensors],
    *,
    max_length: int,
) -> TokenAttributeTargetTensors:
    return TokenAttributeTargetTensors(
        kind_ids=_pad_attribute_target([target.kind_ids for target in targets], max_length=max_length),
        degree_ids=_pad_attribute_target([target.degree_ids for target in targets], max_length=max_length),
        accidental_ids=_pad_attribute_target([target.accidental_ids for target in targets], max_length=max_length),
        octave_offset_ids=_pad_attribute_target(
            [target.octave_offset_ids for target in targets],
            max_length=max_length,
        ),
        duration_ids=_pad_attribute_target([target.duration_ids for target in targets], max_length=max_length),
        hand_ids=_pad_attribute_target([target.hand_ids for target in targets], max_length=max_length),
    )


def _tensor_from_attribute_values(values: list[int]) -> Tensor:
    return torch.tensor(values, dtype=torch.long)


def _pad_attribute_target(targets: list[Tensor], *, max_length: int) -> Tensor:
    padded = torch.full((len(targets), max_length), ABSENT_ATTRIBUTE_ID, dtype=torch.long)
    for row_index, target in enumerate(targets):
        if target.ndim != 1:
            raise ValueError(f"attribute target tensors must be 1D, got {target.ndim}D")

        length = target.size(0)
        if length > max_length:
            raise ValueError(f"attribute target length {length} exceeds max_length={max_length}")

        padded[row_index, :length] = target

    return padded
