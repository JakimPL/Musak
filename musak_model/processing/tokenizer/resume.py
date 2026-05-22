from __future__ import annotations

import logging
from pathlib import Path

from musak_model.processing.parser import ParsedScoreArtifact
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.profiler import ProcessingProfilerProtocol
from musak_model.processing.snapshot import TokenizerSnapshot
from musak_model.processing.tokenizer.output import (
    clear_tokenization_outputs,
    truncate_manifest_rows,
    truncate_text_lines,
)
from musak_model.processing.tokenizer.state import (
    TokenizationResumeState,
    append_tokenization_state_header,
    load_tokenization_resume_state,
    resume_state_outputs_missing,
)

_LOGGER = logging.getLogger(__name__)


class TokenizationOutputPaths:
    def __init__(
        self,
        *,
        encoded_jsonl_path: Path,
        encoded_manifest_path: Path,
        tokenizer_snapshot_path: Path,
        state_path: Path,
    ) -> None:
        self.encoded_jsonl_path = encoded_jsonl_path
        self.encoded_manifest_path = encoded_manifest_path
        self.tokenizer_snapshot_path = tokenizer_snapshot_path
        self.state_path = state_path

    @classmethod
    def from_paths(
        cls,
        paths: ProcessedDatasetPaths,
        *,
        snapshot: TokenizerSnapshot,
    ) -> TokenizationOutputPaths:
        return cls(
            encoded_jsonl_path=paths.encoded_jsonl_path(snapshot.tokenizer_hash),
            encoded_manifest_path=paths.encoded_manifest_path(snapshot.tokenizer_hash),
            tokenizer_snapshot_path=paths.tokenizer_snapshot_path(snapshot.tokenizer_hash),
            state_path=paths.tokenization_state_path(snapshot.tokenizer_hash),
        )


def prepare_resume_state(
    output_paths: TokenizationOutputPaths,
    *,
    state_key: str,
    overwrite: bool,
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    profiler: ProcessingProfilerProtocol,
) -> TokenizationResumeState:
    if overwrite:
        clear_outputs(output_paths)

    resume_state = valid_resume_state(output_paths, state_key=state_key)
    if not complete_outputs_exist(parsed_scores=parsed_scores, output_paths=output_paths, resume_state=resume_state):
        with profiler.measure("prepare_encoded_outputs"):
            prepare_outputs_for_resume(output_paths, state_key=state_key, resume_state=resume_state)

    return load_tokenization_resume_state(output_paths.state_path, state_key=state_key)


def valid_resume_state(
    output_paths: TokenizationOutputPaths,
    *,
    state_key: str,
) -> TokenizationResumeState:
    resume_state = load_tokenization_resume_state(output_paths.state_path, state_key=state_key)
    if not resume_state.state_key_matches:
        _LOGGER.warning("Ignoring stale tokenization resume state for %s", output_paths.state_path.parent)
        clear_outputs(output_paths)
        return load_tokenization_resume_state(output_paths.state_path, state_key=state_key)

    if resume_state_outputs_missing(
        resume_state=resume_state,
        encoded_jsonl_path=output_paths.encoded_jsonl_path,
        encoded_manifest_path=output_paths.encoded_manifest_path,
    ):
        _LOGGER.warning(
            "Ignoring tokenization resume state with missing outputs for %s", output_paths.state_path.parent
        )
        clear_outputs(output_paths)
        return load_tokenization_resume_state(output_paths.state_path, state_key=state_key)

    return resume_state


def prepare_outputs_for_resume(
    output_paths: TokenizationOutputPaths,
    *,
    state_key: str,
    resume_state: TokenizationResumeState,
) -> None:
    if not output_paths.state_path.exists():
        clear_outputs_without_snapshot(output_paths)
        append_tokenization_state_header(output_paths.state_path, state_key=state_key)
        return

    truncate_text_lines(output_paths.encoded_jsonl_path, resume_state.encoded_line_count)
    truncate_manifest_rows(output_paths.encoded_manifest_path, resume_state.manifest_row_count)


def complete_outputs_exist(
    *,
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    output_paths: TokenizationOutputPaths,
    resume_state: TokenizationResumeState,
) -> bool:
    if not (
        output_paths.encoded_jsonl_path.exists()
        and output_paths.encoded_manifest_path.exists()
        and output_paths.tokenizer_snapshot_path.exists()
    ):
        return False

    source_ids = {artifact.source_id_value for artifact in parsed_scores}
    return source_ids <= resume_state.completed_source_ids or not resume_state.completed_source_ids


def clear_outputs(output_paths: TokenizationOutputPaths) -> None:
    clear_tokenization_outputs(
        encoded_jsonl_path=output_paths.encoded_jsonl_path,
        encoded_manifest_path=output_paths.encoded_manifest_path,
        tokenizer_snapshot_path=output_paths.tokenizer_snapshot_path,
        state_path=output_paths.state_path,
    )


def clear_outputs_without_snapshot(output_paths: TokenizationOutputPaths) -> None:
    clear_tokenization_outputs(
        encoded_jsonl_path=output_paths.encoded_jsonl_path,
        encoded_manifest_path=output_paths.encoded_manifest_path,
        tokenizer_snapshot_path=None,
        state_path=output_paths.state_path,
    )
