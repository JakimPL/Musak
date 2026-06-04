from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from torch import Tensor

from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.conditioning.harmony.vocabulary import (
    CADENCE_STRENGTH_VOCABULARY_SIZE,
    CHORD_EXTENSION_VOCABULARY_SIZE,
    CHORD_QUALITY_VOCABULARY_SIZE,
    HARMONIC_FUNCTION_VOCABULARY_SIZE,
    ROOT_DEGREE_VOCABULARY_SIZE,
)


class HarmonicPlanReconstructionFieldName(StrEnum):
    HARMONIC_FUNCTION = "harmonic_function"
    ROOT_DEGREE = "root_degree"
    QUALITY = "quality"
    EXTENSION = "extension"
    CADENCE_STRENGTH = "cadence_strength"


@dataclass(frozen=True)
class HarmonicPlanReconstructionField:
    name: HarmonicPlanReconstructionFieldName
    vocabulary_size: int


HARMONIC_PLAN_RECONSTRUCTION_FIELDS: Final[tuple[HarmonicPlanReconstructionField, ...]] = (
    HarmonicPlanReconstructionField(
        HarmonicPlanReconstructionFieldName.HARMONIC_FUNCTION,
        HARMONIC_FUNCTION_VOCABULARY_SIZE,
    ),
    HarmonicPlanReconstructionField(HarmonicPlanReconstructionFieldName.ROOT_DEGREE, ROOT_DEGREE_VOCABULARY_SIZE),
    HarmonicPlanReconstructionField(HarmonicPlanReconstructionFieldName.QUALITY, CHORD_QUALITY_VOCABULARY_SIZE),
    HarmonicPlanReconstructionField(HarmonicPlanReconstructionFieldName.EXTENSION, CHORD_EXTENSION_VOCABULARY_SIZE),
    HarmonicPlanReconstructionField(
        HarmonicPlanReconstructionFieldName.CADENCE_STRENGTH,
        CADENCE_STRENGTH_VOCABULARY_SIZE,
    ),
)


def harmonic_plan_reconstruction_target_tensor(
    harmonic_plan: HarmonicPlanInputTensors,
    field_name: HarmonicPlanReconstructionFieldName,
) -> Tensor:
    match field_name:
        case HarmonicPlanReconstructionFieldName.HARMONIC_FUNCTION:
            return harmonic_plan.harmonic_function_ids
        case HarmonicPlanReconstructionFieldName.ROOT_DEGREE:
            return harmonic_plan.root_degree_ids
        case HarmonicPlanReconstructionFieldName.QUALITY:
            return harmonic_plan.quality_ids
        case HarmonicPlanReconstructionFieldName.EXTENSION:
            return harmonic_plan.extension_ids
        case HarmonicPlanReconstructionFieldName.CADENCE_STRENGTH:
            return harmonic_plan.cadence_strength_ids
