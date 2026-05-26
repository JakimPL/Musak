from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from musak_model.evaluation.diagnostics import SegmentDiagnostics
from musak_model.evaluation.generation.schema import GenerationSample


def suite_metrics(suite_name: str, samples: list[GenerationSample]) -> dict[str, float]:
    prefix = f"generation/{suite_name}"
    if not samples:
        return {f"{prefix}/count/samples": 0.0}

    diagnostics = [sample.diagnostics for sample in samples if sample.diagnostics is not None]
    metrics = {
        **_sample_outcome_metrics(prefix, samples),
        **_constraint_metrics(prefix, samples),
        **_bar_completion_metrics(prefix, samples),
        **_activity_metrics(prefix, diagnostics),
        **_token_metrics(prefix, diagnostics),
        **_pitch_and_rhythm_metrics(prefix, diagnostics),
        **_playability_metrics(prefix, diagnostics),
        **_coordination_metrics(prefix, diagnostics),
    }
    return {name: value for name, value in metrics.items() if math.isfinite(value)}


def _sample_outcome_metrics(prefix: str, samples: list[GenerationSample]) -> dict[str, float]:
    return {
        f"{prefix}/count/samples": float(len(samples)),
        f"{prefix}/rate/end": rate(samples, lambda sample: sample.reached_end),
        f"{prefix}/rate/decode_error": rate(samples, lambda sample: sample.decode_error is not None),
        f"{prefix}/mean/generated_tokens": mean(sample.generated_token_count for sample in samples),
    }


def _constraint_metrics(prefix: str, samples: list[GenerationSample]) -> dict[str, float]:
    return {
        f"{prefix}/rate/constraint_error": rate(samples, lambda sample: sample.constraint_error is not None),
        f"{prefix}/rate/constraint_failure": rate(samples, lambda sample: sample.constraint_report.failed),
        f"{prefix}/mean/constraint_valid_token_fraction": mean(
            sample.constraint_report.valid_token_fraction for sample in samples
        ),
        f"{prefix}/mean/constraint_first_failure_step": mean(
            float(sample.constraint_report.first_failure_step)
            for sample in samples
            if sample.constraint_report.first_failure_step is not None
        ),
    }


def _bar_completion_metrics(prefix: str, samples: list[GenerationSample]) -> dict[str, float]:
    return {
        f"{prefix}/rate/target_bar_completion": rate(
            samples,
            lambda sample: sample.reached_end and sample.completed_bars == sample.target_bar_count,
        ),
        f"{prefix}/mean/bar_count_error": mean(
            abs(sample.completed_bars - sample.target_bar_count) for sample in samples
        ),
        f"{prefix}/mean/completed_bars": mean(sample.completed_bars for sample in samples),
    }


def _activity_metrics(prefix: str, diagnostics: list[SegmentDiagnostics]) -> dict[str, float]:
    return {
        f"{prefix}/rate/empty_score": rate(diagnostics, lambda item: item.empty_score),
        f"{prefix}/rate/one_hand_only": rate(diagnostics, lambda item: item.one_hand_only),
        f"{prefix}/mean/right_silence_fraction": mean(item.right_silence_fraction for item in diagnostics),
        f"{prefix}/mean/left_silence_fraction": mean(item.left_silence_fraction for item in diagnostics),
        f"{prefix}/mean/both_hands_silence_fraction": mean(item.both_hands_silence_fraction for item in diagnostics),
        f"{prefix}/mean/both_hands_active_fraction": mean(item.both_hands_active_fraction for item in diagnostics),
        f"{prefix}/mean/hand_activity_balance": mean(item.hand_activity_balance for item in diagnostics),
        f"{prefix}/mean/silent_bar_count": mean(item.silent_bar_count for item in diagnostics),
        f"{prefix}/mean/silent_bar_fraction": mean(item.silent_bar_fraction for item in diagnostics),
        f"{prefix}/mean/silent_edge_bar_count": mean(item.silent_edge_bar_count for item in diagnostics),
    }


def _token_metrics(prefix: str, diagnostics: list[SegmentDiagnostics]) -> dict[str, float]:
    return {
        f"{prefix}/mean/note_token_fraction": mean(item.note_token_fraction for item in diagnostics),
        f"{prefix}/mean/rest_token_fraction": mean(item.rest_token_fraction for item in diagnostics),
        f"{prefix}/mean/hold_token_fraction": mean(item.hold_token_fraction for item in diagnostics),
    }


def _pitch_and_rhythm_metrics(prefix: str, diagnostics: list[SegmentDiagnostics]) -> dict[str, float]:
    return {
        f"{prefix}/mean/accidental_note_fraction": mean(item.accidental_note_fraction for item in diagnostics),
        f"{prefix}/mean/in_scale_note_fraction": mean(item.in_scale_note_fraction for item in diagnostics),
        f"{prefix}/mean/note_density_per_beat": mean(item.note_density_per_beat for item in diagnostics),
        f"{prefix}/mean/onset_density_per_beat": mean(item.onset_density_per_beat for item in diagnostics),
        f"{prefix}/mean/right_onset_density_per_beat": mean(item.right_onset_density_per_beat for item in diagnostics),
        f"{prefix}/mean/left_onset_density_per_beat": mean(item.left_onset_density_per_beat for item in diagnostics),
        f"{prefix}/mean/shortest_note_duration_beats": mean(item.shortest_note_duration_beats for item in diagnostics),
        f"{prefix}/rate/has_dotted_notes": rate(diagnostics, lambda item: item.has_dotted_notes),
    }


def _playability_metrics(prefix: str, diagnostics: list[SegmentDiagnostics]) -> dict[str, float]:
    return {
        f"{prefix}/mean/max_notes_per_onset": mean(item.max_notes_per_onset for item in diagnostics),
        f"{prefix}/mean/max_notes_per_hand": mean(item.max_notes_per_hand for item in diagnostics),
        f"{prefix}/mean/max_onset_span_semitones": mean(item.max_onset_span_semitones for item in diagnostics),
        f"{prefix}/mean/max_melodic_gap_semitones": mean(item.max_melodic_gap_semitones for item in diagnostics),
        f"{prefix}/mean/static_hand_span_degrees": mean(item.static_hand_span_degrees for item in diagnostics),
    }


def _coordination_metrics(prefix: str, diagnostics: list[SegmentDiagnostics]) -> dict[str, float]:
    return {
        f"{prefix}/mean/synchronized_onset_fraction": mean(item.synchronized_onset_fraction for item in diagnostics),
        f"{prefix}/mean/independent_onset_fraction": mean(item.independent_onset_fraction for item in diagnostics),
    }


def rate[T](items: list[T], predicate: Callable[[T], bool]) -> float:
    if not items:
        return math.nan

    return sum(bool(predicate(item)) for item in items) / len(items)


def mean(values: Iterable[float | int]) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return math.nan

    return sum(collected) / len(collected)
