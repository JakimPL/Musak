from __future__ import annotations

from dataclasses import dataclass

from musak_model.evaluation.diagnostics import SegmentDiagnostics
from musak_model.tokens.schema import Token


@dataclass(frozen=True)
class ConstraintReport:
    failed: bool
    valid_token_fraction: float
    first_failure_step: int | None
    error: str | None


@dataclass(frozen=True)
class GenerationSample:
    tokens: list[Token]
    reached_end: bool
    generated_token_count: int
    constraint_error: str | None
    constraint_report: ConstraintReport
    diagnostics: SegmentDiagnostics | None
    decode_error: str | None
    completed_bars: int
    target_bar_count: int


@dataclass(frozen=True)
class GenerationSampleSuite:
    name: str
    samples: list[GenerationSample]


@dataclass(frozen=True)
class GenerationEvaluationResult:
    metrics: dict[str, float]
    sample_suites: tuple[GenerationSampleSuite, ...]
