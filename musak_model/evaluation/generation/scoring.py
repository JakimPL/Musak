from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from musak_model.evaluation.diagnostics import SegmentDiagnostics
from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.schema import GenerationSample

_METRIC_TOTAL_NAME: Final[str] = "sample_penalty"
_COUNT_SCORED_SAMPLES_NAME: Final[str] = "scored_samples"

type _SamplePenaltyFunction = Callable[[GenerationSample, GenerationEvaluationOptions], float | None]
type _DiagnosticPenaltyFunction = Callable[[SegmentDiagnostics, GenerationEvaluationOptions], float | None]


@dataclass(frozen=True)
class GenerationSampleScoreTerm:
    name: str
    penalty: float


@dataclass(frozen=True)
class GenerationSampleScore:
    total_penalty: float
    terms: tuple[GenerationSampleScoreTerm, ...]


@dataclass(frozen=True)
class _SamplePenaltySpec:
    name: str
    penalty: _SamplePenaltyFunction


@dataclass(frozen=True)
class _DiagnosticPenaltySpec:
    name: str
    penalty: _DiagnosticPenaltyFunction


_SAMPLE_PENALTY_SPECS: Final[tuple[_SamplePenaltySpec, ...]] = (
    _SamplePenaltySpec("decode_error", lambda sample, _config: float(sample.decode_error is not None)),
    _SamplePenaltySpec("constraint_error", lambda sample, _config: float(sample.constraint_error is not None)),
    _SamplePenaltySpec("constraint_failure", lambda sample, _config: float(sample.constraint_report.failed)),
    _SamplePenaltySpec("incomplete", lambda sample, _config: float(not sample.reached_end)),
    _SamplePenaltySpec("bar_count_error", lambda sample, _config: _normalized_count_error(sample)),
    _SamplePenaltySpec(
        "invalid_token_fraction",
        lambda sample, _config: 1.0 - sample.constraint_report.valid_token_fraction,
    ),
)

_DIAGNOSTIC_PENALTY_SPECS: Final[tuple[_DiagnosticPenaltySpec, ...]] = (
    _DiagnosticPenaltySpec("empty_score", lambda diagnostics, _config: float(diagnostics.empty_score)),
    _DiagnosticPenaltySpec("one_hand_only", lambda diagnostics, _config: float(diagnostics.one_hand_only)),
    _DiagnosticPenaltySpec("silent_bar_fraction", lambda diagnostics, _config: diagnostics.silent_bar_fraction),
    _DiagnosticPenaltySpec(
        "both_hands_silence_fraction",
        lambda diagnostics, _config: diagnostics.both_hands_silence_fraction,
    ),
    _DiagnosticPenaltySpec(
        "hand_activity_imbalance",
        lambda diagnostics, _config: 1.0 - diagnostics.hand_activity_balance,
    ),
    _DiagnosticPenaltySpec(
        "out_of_scale_note_fraction",
        lambda diagnostics, _config: 1.0 - diagnostics.in_scale_note_fraction,
    ),
    _DiagnosticPenaltySpec(
        "accidental_note_fraction",
        lambda diagnostics, _config: diagnostics.accidental_note_fraction,
    ),
    _DiagnosticPenaltySpec(
        "dotted_duration_when_disallowed",
        lambda diagnostics, config: float(diagnostics.has_dotted_notes and not config.allow_dotted_durations),
    ),
    _DiagnosticPenaltySpec(
        "minimum_duration_violation",
        lambda diagnostics, config: _minimum_duration_violation(diagnostics, config=config),
    ),
    _DiagnosticPenaltySpec(
        "max_notes_per_hand_excess",
        lambda diagnostics, config: _optional_limit_excess(diagnostics.max_notes_per_hand, config.max_notes_per_hand),
    ),
    _DiagnosticPenaltySpec(
        "max_onset_span_excess",
        lambda diagnostics, config: _optional_limit_excess(
            diagnostics.max_onset_span_semitones,
            config.maximum_onset_span_semitones,
        ),
    ),
    _DiagnosticPenaltySpec(
        "max_melodic_gap_excess",
        lambda diagnostics, config: _optional_limit_excess(
            diagnostics.max_melodic_gap_semitones,
            config.maximum_pitch_gap_semitones,
        ),
    ),
    _DiagnosticPenaltySpec(
        "static_hand_span_excess",
        lambda diagnostics, config: _optional_limit_excess(
            diagnostics.static_hand_span_degrees,
            config.maximum_static_hand_span_degrees,
        ),
    ),
)


def generation_sample_score(
    sample: GenerationSample,
    *,
    config: GenerationEvaluationOptions,
) -> GenerationSampleScore:
    terms = [
        GenerationSampleScoreTerm(name=spec.name, penalty=penalty)
        for spec in _SAMPLE_PENALTY_SPECS
        if (penalty := spec.penalty(sample, config)) is not None
    ]
    if sample.diagnostics is not None:
        terms.extend(
            GenerationSampleScoreTerm(name=spec.name, penalty=penalty)
            for spec in _DIAGNOSTIC_PENALTY_SPECS
            if (penalty := spec.penalty(sample.diagnostics, config)) is not None
        )

    return GenerationSampleScore(
        total_penalty=sum(term.penalty for term in terms),
        terms=tuple(terms),
    )


def generation_sample_score_metrics(
    suite_name: str,
    samples: list[GenerationSample],
    *,
    config: GenerationEvaluationOptions,
) -> dict[str, float]:
    prefix = f"generation/{suite_name}"
    if not samples:
        return {f"{prefix}/count/{_COUNT_SCORED_SAMPLES_NAME}": 0.0}

    scores = [generation_sample_score(sample, config=config) for sample in samples]
    metrics = {
        f"{prefix}/count/{_COUNT_SCORED_SAMPLES_NAME}": float(len(scores)),
        f"{prefix}/mean/{_METRIC_TOTAL_NAME}": _mean(score.total_penalty for score in scores),
    }
    term_penalties: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        for term in score.terms:
            term_penalties[term.name].append(term.penalty)

    metrics.update(
        {
            f"{prefix}/mean/{_METRIC_TOTAL_NAME}_{term_name}": _mean(penalties)
            for term_name, penalties in term_penalties.items()
        }
    )
    return metrics


def _normalized_count_error(sample: GenerationSample) -> float:
    return abs(sample.completed_bars - sample.target_bar_count) / max(sample.target_bar_count, 1)


def _minimum_duration_violation(
    diagnostics: SegmentDiagnostics,
    *,
    config: GenerationEvaluationOptions,
) -> float | None:
    if config.minimum_duration_denominator is None or diagnostics.shortest_note_duration_beats == 0.0:
        return None

    minimum_duration_beats = float(Fraction(1, config.minimum_duration_denominator) * config.time_denominator)
    return max(0.0, minimum_duration_beats - diagnostics.shortest_note_duration_beats) / minimum_duration_beats


def _optional_limit_excess(value: int, limit: int | None) -> float | None:
    if limit is None:
        return None

    return max(0.0, value - limit) / max(limit, 1)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return math.nan

    return sum(collected) / len(collected)
