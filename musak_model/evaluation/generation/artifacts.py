from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from musak_model.conditioning.harmony.planner import HarmonicPlan, HarmonicPlanAlternative
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.decoder.music21 import write_segment
from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.sampling import segment_from_tokens
from musak_model.evaluation.generation.schema import ConstraintReport, GenerationEvaluationResult, GenerationSample
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.text import tokens_to_text

SAMPLES_MANIFEST_NAME: Final[str] = "samples.jsonl"
TOKEN_TEXT_SUFFIX: Final[str] = ".tokens.txt"
MUSICXML_SUFFIX: Final[str] = ".musicxml"


@dataclass(frozen=True)
class GenerationSampleArtifactPaths:
    token_text_path: Path
    musicxml_path: Path | None


def write_generation_sample_artifacts(
    result: GenerationEvaluationResult,
    *,
    output_directory: Path,
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []
    for suite in result.sample_suites:
        suite_directory = output_directory / suite.name
        suite_directory.mkdir(parents=True, exist_ok=True)
        for sample_index, sample in enumerate(suite.samples):
            paths = _write_sample_artifacts(
                sample,
                sample_index=sample_index,
                suite_directory=suite_directory,
                config=config,
                duration_vocabulary=duration_vocabulary,
            )
            manifest_lines.append(
                json.dumps(
                    _manifest_record(
                        sample,
                        suite_name=suite.name,
                        sample_index=sample_index,
                        paths=paths,
                        output_directory=output_directory,
                    ),
                    sort_keys=True,
                )
            )

    manifest_text = "\n".join(manifest_lines)
    if manifest_text:
        manifest_text += "\n"

    (output_directory / SAMPLES_MANIFEST_NAME).write_text(manifest_text, encoding="utf-8")


def _write_sample_artifacts(
    sample: GenerationSample,
    *,
    sample_index: int,
    suite_directory: Path,
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
) -> GenerationSampleArtifactPaths:
    sample_stem = f"sample_{sample_index:04d}"
    token_text_path = suite_directory / f"{sample_stem}{TOKEN_TEXT_SUFFIX}"
    token_text_path.write_text(
        tokens_to_text(sample.tokens, duration_vocabulary=duration_vocabulary) + "\n",
        encoding="utf-8",
    )

    if sample.decode_error is not None:
        return GenerationSampleArtifactPaths(token_text_path=token_text_path, musicxml_path=None)

    segment = segment_from_tokens(sample.tokens, config=config)
    musicxml_path = suite_directory / f"{sample_stem}{MUSICXML_SUFFIX}"
    written_musicxml_path = write_segment(
        segment,
        duration_vocabulary=duration_vocabulary,
        path=musicxml_path,
        format_name="musicxml",
    )
    return GenerationSampleArtifactPaths(token_text_path=token_text_path, musicxml_path=written_musicxml_path)


def _manifest_record(
    sample: GenerationSample,
    *,
    suite_name: str,
    sample_index: int,
    paths: GenerationSampleArtifactPaths,
    output_directory: Path,
) -> dict[str, object]:
    return {
        "suite": suite_name,
        "sample_index": sample_index,
        "token_text_path": _relative_path_text(paths.token_text_path, output_directory),
        "musicxml_path": _optional_relative_path_text(paths.musicxml_path, output_directory),
        "reached_end": sample.reached_end,
        "generated_token_count": sample.generated_token_count,
        "constraint_error": sample.constraint_error,
        "constraint_report": _constraint_report_values(sample.constraint_report),
        "diagnostics": None if sample.diagnostics is None else sample.diagnostics.to_manifest_values(),
        "decode_error": sample.decode_error,
        "harmonic_plan": _optional_harmonic_plan_values(sample.harmonic_plan_windows),
        "harmonic_plan_summary": _optional_harmonic_plan_summary(sample.harmonic_plan),
        "harmonic_plan_alternatives": _optional_harmonic_plan_alternatives(sample.harmonic_plan),
        "completed_bars": sample.completed_bars,
        "target_bar_count": sample.target_bar_count,
    }


def _constraint_report_values(report: ConstraintReport) -> dict[str, float | int | bool | str | None]:
    return {
        "failed": report.failed,
        "valid_token_fraction": report.valid_token_fraction,
        "first_failure_step": report.first_failure_step,
        "error": report.error,
    }


def _optional_harmonic_plan_values(
    windows: tuple[HarmonicPlanWindow, ...] | None,
) -> list[dict[str, object]] | None:
    if windows is None:
        return None

    return [
        {
            "start": str(window.start),
            "end": str(window.end),
            "root_degree": window.chord.root_degree,
            "root_accidental": window.chord.root_accidental,
            "quality": window.chord.quality.value,
            "extension": window.chord.extension.value,
            "harmonic_function": None if window.harmonic_function is None else window.harmonic_function.value,
            "slot_role": None if window.slot_role is None else window.slot_role.value,
            "distance_to_end": window.distance_to_end,
            "cadence_strength": window.cadence_strength,
            "tension_level": window.tension_level,
            "plan_confidence": window.plan_confidence,
            "score_terms": dict(window.score_terms),
        }
        for window in windows
    ]


def _optional_harmonic_plan_summary(plan: HarmonicPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None

    return {
        "score": plan.score,
        "window_count": len(plan.windows),
        "alternative_count": len(plan.alternatives),
        "slot_roles": [None if window.slot_role is None else window.slot_role.value for window in plan.windows],
        "distance_to_end": [window.distance_to_end for window in plan.windows],
        "final_harmonic_function": _final_harmonic_function(plan.windows),
        "final_root_degree": None if not plan.windows else plan.windows[-1].chord.root_degree,
        "unique_chord_count": _unique_chord_count(plan.windows),
        "longest_same_chord_run": _longest_same_chord_run(plan.windows),
        "top_alternative_scores": [alternative.score for alternative in plan.alternatives],
    }


def _optional_harmonic_plan_alternatives(plan: HarmonicPlan | None) -> list[dict[str, object]] | None:
    if plan is None:
        return None

    return [_harmonic_plan_alternative_values(alternative) for alternative in plan.alternatives]


def _harmonic_plan_alternative_values(alternative: HarmonicPlanAlternative) -> dict[str, object]:
    return {
        "score": alternative.score,
        "windows": _optional_harmonic_plan_values(alternative.windows),
    }


def _final_harmonic_function(windows: tuple[HarmonicPlanWindow, ...]) -> str | None:
    if not windows or windows[-1].harmonic_function is None:
        return None

    return windows[-1].harmonic_function.value


def _unique_chord_count(windows: tuple[HarmonicPlanWindow, ...]) -> int:
    return len({window.chord for window in windows})


def _longest_same_chord_run(windows: tuple[HarmonicPlanWindow, ...]) -> int:
    longest_run = 0
    current_run = 0
    previous_window: HarmonicPlanWindow | None = None
    for window in windows:
        if previous_window is not None and window.chord == previous_window.chord:
            current_run += 1
        else:
            current_run = 1

        longest_run = max(longest_run, current_run)
        previous_window = window

    return longest_run


def _optional_relative_path_text(path: Path | None, base_directory: Path) -> str | None:
    if path is None:
        return None

    return _relative_path_text(path, base_directory)


def _relative_path_text(path: Path, base_directory: Path) -> str:
    return path.relative_to(base_directory).as_posix()
