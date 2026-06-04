from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.conditioning.config import HarmonicConditioningConfig, HarmonicFusionMode
from musak_model.conditioning.harmony.fields import (
    HARMONIC_PLAN_TENSOR_FIELDS,
    HarmonicPlanTensorField,
    harmonic_plan_tensor,
)
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.conditioning.harmony.vocabulary import HARMONIC_PLAN_UNKNOWN_ID


@dataclass(frozen=True)
class HarmonicPlanConditioningOutput:
    embedding_delta: Tensor
    plan_embeddings: Tensor | None = None
    gate_values: Tensor | None = None


class HarmonicPlanConditioning(nn.Module):
    def __init__(
        self,
        config: HarmonicConditioningConfig,
        *,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self._config = config
        self._hidden_size = hidden_size
        self._embeddings = HarmonicPlanEmbeddings(hidden_size=hidden_size)
        self._field_dropout = nn.Dropout(config.plan_field_dropout)
        self._plan_encoder: nn.TransformerEncoder | None = None
        self._gate: nn.Linear | None = None
        if config.fusion == HarmonicFusionMode.GATED_RESIDUAL:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.plan_encoder_heads,
                dim_feedforward=hidden_size * 4,
                dropout=config.plan_encoder_dropout,
                batch_first=True,
            )
            self._plan_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.plan_encoder_layers)
            self._gate = nn.Linear(hidden_size * 2, hidden_size)
            nn.init.constant_(self._gate.bias, config.gate_init_bias)

    def forward(
        self,
        harmonic_plan: HarmonicPlanInputTensors | None,
        *,
        token_embeddings: Tensor,
        token_ids: Tensor,
        token_padding_mask: Tensor | None,
    ) -> HarmonicPlanConditioningOutput:
        plan_embeddings = self._embeddings(
            harmonic_plan,
            token_ids=token_ids,
            dtype=token_embeddings.dtype,
        )
        if harmonic_plan is None:
            return HarmonicPlanConditioningOutput(embedding_delta=plan_embeddings)

        plan_embeddings = self._field_dropout(plan_embeddings)
        match self._config.fusion:
            case HarmonicFusionMode.ADDITIVE:
                return HarmonicPlanConditioningOutput(
                    embedding_delta=plan_embeddings,
                    plan_embeddings=plan_embeddings,
                )
            case HarmonicFusionMode.GATED_RESIDUAL:
                return self._gated_residual_output(
                    plan_embeddings,
                    token_embeddings=token_embeddings,
                    token_padding_mask=token_padding_mask,
                )

    def _gated_residual_output(
        self,
        plan_embeddings: Tensor,
        *,
        token_embeddings: Tensor,
        token_padding_mask: Tensor | None,
    ) -> HarmonicPlanConditioningOutput:
        if self._plan_encoder is None or self._gate is None:
            raise ValueError("gated harmonic fusion requires a plan encoder and gate")

        plan_context = self._plan_encoder(
            plan_embeddings,
            src_key_padding_mask=token_padding_mask,
        )
        gate_values = torch.sigmoid(self._gate(torch.cat((token_embeddings, plan_embeddings), dim=-1)))
        return HarmonicPlanConditioningOutput(
            embedding_delta=self._config.harmony_adherence_alpha * gate_values * plan_context,
            plan_embeddings=plan_embeddings,
            gate_values=gate_values,
        )


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
