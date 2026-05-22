from dataclasses import dataclass
from pathlib import Path

from musak_model.data.config import SegmentationConfig
from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.processing.config import TokenizationProcessingConfig
from musak_model.processing.parser import ParsedScoreArtifact
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.tokens.config import TokenizationConfig


@dataclass(frozen=True)
class TokenizeDatasetResult:
    parsed_manifest_path: Path
    encoded_manifest_path: Path
    tokenizer_snapshot_path: Path
    encoded_count: int
    segment_count: int
    scale_matcher_config: ScaleMatcherConfig


@dataclass(frozen=True)
class TokenizedSourceResult:
    source_id_value: str
    temp_encoded_jsonl_path: Path
    temp_encoded_manifest_path: Path
    encoded_count: int
    manifest_row_count: int


@dataclass(frozen=True)
class TokenizationBatchTask:
    index: int
    artifacts: tuple[ParsedScoreArtifact, ...]
    dataset_root: Path
    paths: ProcessedDatasetPaths
    final_encoded_jsonl_path: Path
    temp_root: Path
    segmentation_config: SegmentationConfig
    tokenization_config: TokenizationConfig
    tokenization_processing_config: TokenizationProcessingConfig
    difficulty_labels: dict[str, int | None] | None


@dataclass(frozen=True)
class TokenizationBatchResult:
    index: int
    source_results: tuple[TokenizedSourceResult, ...]
