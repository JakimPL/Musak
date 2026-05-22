from dataclasses import dataclass

from musak_model.data.schema import ScaleMatchDiagnostics
from musak_model.tokens.schema import ScaleType


@dataclass(frozen=True)
class ScaleMatch:
    scale_root: int
    scale_type: ScaleType
    diagnostics: ScaleMatchDiagnostics


@dataclass(frozen=True)
class ScaleCandidate:
    scale_root: int
    scale_type: ScaleType
    in_scale_weight_fraction: float
    pitch_classes: frozenset[int]


@dataclass(frozen=True)
class CandidateExplanation:
    candidate: ScaleCandidate
    explained_out_of_scale_weight_fraction: float
    unexplained_out_of_scale_weight_fraction: float
    explanation_pitch_class_count: int
    support_candidate_count: int
