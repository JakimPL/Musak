from dataclasses import dataclass
from pathlib import Path

from musak_model.data.schema import ParsedScore
from musak_model.processing.paths import ProcessedDatasetPaths


@dataclass(frozen=True)
class ParsedScoreArtifact:
    source_id_value: str
    source_path: Path
    parsed_path: Path
    score: ParsedScore


@dataclass(frozen=True)
class ParseDatasetResult:
    parsed_manifest_path: Path
    parsed_count: int
    error_count: int
    parsed_scores: tuple[ParsedScoreArtifact, ...]


@dataclass(frozen=True)
class ParsedScoreTask:
    index: int
    source_path: Path
    dataset_root: Path
    paths: ProcessedDatasetPaths
    overwrite: bool


@dataclass(frozen=True)
class ParsedScoreResult:
    index: int
    source_id_value: str
    source_path: Path
    parsed_path: Path
    row: dict[str, object]
    score: ParsedScore | None
