from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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


def _optional_relative_path_text(path: Path | None, base_directory: Path) -> str | None:
    if path is None:
        return None

    return _relative_path_text(path, base_directory)


def _relative_path_text(path: Path, base_directory: Path) -> str:
    return path.relative_to(base_directory).as_posix()
