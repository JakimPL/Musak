from dataclasses import dataclass
from pathlib import Path

from musak_model.data.config import (
    DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
    DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
    DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
    DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
)


@dataclass(frozen=True)
class TokenizeDatasetResult:
    parsed_manifest_path: Path
    encoded_manifest_path: Path
    tokenizer_snapshot_path: Path
    encoded_count: int
    segment_count: int
    scale_match_support_score_margin: float = DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN
    scale_match_selection_score_margin: float = DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN
    scale_match_maximum_unexplained_weight_fraction: float = DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION
    scale_match_maximum_explanation_pitch_class_count: int = DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT
