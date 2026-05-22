from dataclasses import dataclass
from pathlib import Path

from musak_model.data.scale_matcher.config import ScaleMatcherConfig


@dataclass(frozen=True)
class TokenizeDatasetResult:
    parsed_manifest_path: Path
    encoded_manifest_path: Path
    tokenizer_snapshot_path: Path
    encoded_count: int
    segment_count: int
    scale_matcher_config: ScaleMatcherConfig
