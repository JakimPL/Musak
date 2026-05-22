from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingProfileRecord:
    stage: str
    seconds: float
    source_file: str


@dataclass(frozen=True)
class ProcessingProfileSummary:
    total_seconds: float
    stage_totals: dict[str, float]
    stage_counts: dict[str, int]
    stage_means: dict[str, float]
    per_file_totals: dict[str, float]


@dataclass(frozen=True)
class ProcessingProfileStageStats:
    stage: str
    count: int
    total_seconds: float
    mean_seconds: float
    min_seconds: float
    max_seconds: float
