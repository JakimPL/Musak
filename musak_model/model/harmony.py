from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.conditioning.harmony.fields import (
    HARMONIC_PLAN_TENSOR_FIELDS,
    HarmonicPlanTensorField,
    harmonic_plan_tensor,
)
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.conditioning.harmony.vocabulary import HARMONIC_PLAN_UNKNOWN_ID


class HarmonicPlanEmbeddings(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self._embeddings = nn.ModuleDict(
            {
                field.name: nn.Embedding(
                    field.vocabulary_size,
                    hidden_size,
                    padding_idx=HARMONIC_PLAN_UNKNOWN_ID,
                )
                for field in HARMONIC_PLAN_TENSOR_FIELDS
            }
        )

    def forward(
        self,
        harmonic_plan: HarmonicPlanInputTensors | None,
        *,
        token_ids: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        output = torch.zeros(
            (*token_ids.shape, self._hidden_size),
            device=token_ids.device,
            dtype=dtype,
        )
        if harmonic_plan is None:
            return output

        for field in HARMONIC_PLAN_TENSOR_FIELDS:
            field_ids = harmonic_plan_tensor(harmonic_plan, field)
            _validate_harmonic_plan_ids(field_ids, field=field, token_shape=token_ids.shape)
            embedding = cast(nn.Embedding, self._embeddings[field.name])
            output = output + cast(Tensor, embedding(field_ids.to(device=token_ids.device))).to(dtype=dtype)

        return output


def _validate_harmonic_plan_ids(
    field_ids: Tensor,
    *,
    field: HarmonicPlanTensorField,
    token_shape: torch.Size,
) -> None:
    if field_ids.shape != token_shape:
        raise ValueError(
            f"harmonic plan field {field.name} shape {tuple(field_ids.shape)} "
            f"does not match token shape {tuple(token_shape)}"
        )

    if field_ids.ndim != 2:
        raise ValueError(f"harmonic plan field {field.name} must be a 2D tensor")

    if bool(torch.any(field_ids < 0).item()):
        raise ValueError(f"harmonic plan field {field.name} contains a negative id")

    if bool(torch.any(field_ids >= field.vocabulary_size).item()):
        raise ValueError(f"harmonic plan field {field.name} contains an id outside range [0, {field.vocabulary_size})")
