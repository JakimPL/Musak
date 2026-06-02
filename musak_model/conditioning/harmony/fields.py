from dataclasses import dataclass
from typing import Final, cast

from torch import Tensor

from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.conditioning.harmony.vocabulary import (
    CHORD_CHANGE_VOCABULARY_SIZE,
    CHORD_EXTENSION_VOCABULARY_SIZE,
    CHORD_QUALITY_VOCABULARY_SIZE,
    HARMONIC_FUNCTION_VOCABULARY_SIZE,
    ROOT_ACCIDENTAL_VOCABULARY_SIZE,
    ROOT_DEGREE_VOCABULARY_SIZE,
)


@dataclass(frozen=True)
class HarmonicPlanTensorField:
    name: str
    vocabulary_size: int


HARMONIC_PLAN_TENSOR_FIELDS: Final[tuple[HarmonicPlanTensorField, ...]] = (
    HarmonicPlanTensorField("harmonic_function_ids", HARMONIC_FUNCTION_VOCABULARY_SIZE),
    HarmonicPlanTensorField("root_degree_ids", ROOT_DEGREE_VOCABULARY_SIZE),
    HarmonicPlanTensorField("root_accidental_ids", ROOT_ACCIDENTAL_VOCABULARY_SIZE),
    HarmonicPlanTensorField("quality_ids", CHORD_QUALITY_VOCABULARY_SIZE),
    HarmonicPlanTensorField("extension_ids", CHORD_EXTENSION_VOCABULARY_SIZE),
    HarmonicPlanTensorField("chord_change_ids", CHORD_CHANGE_VOCABULARY_SIZE),
)


def harmonic_plan_tensor(plan: HarmonicPlanInputTensors, field: HarmonicPlanTensorField) -> Tensor:
    return cast(Tensor, getattr(plan, field.name))
