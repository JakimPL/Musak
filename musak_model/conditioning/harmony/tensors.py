from collections.abc import Sequence
from typing import Final

import torch
from torch import Tensor

from musak_model.conditioning.harmony.fields import HARMONIC_PLAN_TENSOR_FIELDS, harmonic_plan_tensor
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.conditioning.harmony.vocabulary import HARMONIC_PLAN_UNKNOWN_ID

_HARMONIC_PLAN_PADDING_VALUE: Final[int] = HARMONIC_PLAN_UNKNOWN_ID


def pad_harmonic_plan_input_tensors(
    plans: Sequence[HarmonicPlanInputTensors],
    *,
    max_length: int,
) -> HarmonicPlanInputTensors:
    if not plans:
        raise ValueError("cannot pad empty harmonic plan tensor sequence")

    padded_by_name = {
        field.name: _pad_harmonic_plan_field(
            [harmonic_plan_tensor(plan, field) for plan in plans],
            max_length=max_length,
            field_name=field.name,
        )
        for field in HARMONIC_PLAN_TENSOR_FIELDS
    }
    return HarmonicPlanInputTensors(
        harmonic_function_ids=padded_by_name["harmonic_function_ids"],
        root_degree_ids=padded_by_name["root_degree_ids"],
        root_accidental_ids=padded_by_name["root_accidental_ids"],
        quality_ids=padded_by_name["quality_ids"],
        extension_ids=padded_by_name["extension_ids"],
        chord_change_ids=padded_by_name["chord_change_ids"],
    )


def _pad_harmonic_plan_field(
    tensors: Sequence[Tensor],
    *,
    max_length: int,
    field_name: str,
) -> Tensor:
    output = torch.full((len(tensors), max_length), _HARMONIC_PLAN_PADDING_VALUE, dtype=torch.long)
    for row_index, tensor in enumerate(tensors):
        if tensor.ndim != 1:
            raise ValueError(f"harmonic plan field {field_name} must be a 1D tensor")

        length = tensor.size(0)
        if length > max_length:
            raise ValueError(f"harmonic plan field {field_name} length exceeds max_length={max_length}")

        output[row_index, :length] = tensor.to(dtype=torch.long)

    return output
