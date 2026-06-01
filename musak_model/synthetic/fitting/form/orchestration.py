import logging
from pathlib import Path
from typing import Any

from musak_model.n_grams.profile.chord.schema import ChordDecodeSpec
from musak_model.synthetic.fitting.form.cadence import CadenceDetectionConfig
from musak_model.synthetic.fitting.form.executor import process_missing_form_batches
from musak_model.synthetic.fitting.form.io import (
    FormArtifactPaths,
    write_closing_counts,
    write_histogram_counts,
    write_phrase_length_counts,
    write_segment_length_counts,
)
from musak_model.synthetic.fitting.form.repetition import RepetitionConfig
from musak_model.synthetic.fitting.form.store import FormWorkStore
from musak_model.tokens.config import TokenizationConfig
from musak_shared.files import get_fingerprint

_LOGGER = logging.getLogger(__name__)


def form_state_key(
    *,
    tokenizer_hash: str,
    chord_decode: ChordDecodeSpec,
    cadence_config: CadenceDetectionConfig,
    repetition_config: RepetitionConfig,
    figure_min_n: int,
    figure_max_n: int,
    batch_size: int,
) -> str:
    payload: dict[str, Any] = {
        "tokenizer_hash": tokenizer_hash,
        "chord_decoder": chord_decode.decoder_config.model_dump(mode="json"),
        "chord_vocabulary": chord_decode.vocabulary.model_dump(mode="json"),
        "cadence": cadence_config.model_dump(mode="json"),
        "repetition": repetition_config.model_dump(mode="json"),
        "figure_min_n": figure_min_n,
        "figure_max_n": figure_max_n,
        "batch_size": batch_size,
    }
    return get_fingerprint(payload)


def extract_form_statistics(
    *,
    encoded_jsonl_path: Path,
    artifact_paths: FormArtifactPaths,
    tokenization_config: TokenizationConfig,
    chord_decode: ChordDecodeSpec,
    cadence_config: CadenceDetectionConfig,
    repetition_config: RepetitionConfig,
    figure_min_n: int,
    figure_max_n: int,
    batch_size: int,
    workers: int,
    tokenizer_hash: str,
    show_progress: bool,
    resume: bool,
) -> None:
    state_key = form_state_key(
        tokenizer_hash=tokenizer_hash,
        chord_decode=chord_decode,
        cadence_config=cadence_config,
        repetition_config=repetition_config,
        figure_min_n=figure_min_n,
        figure_max_n=figure_max_n,
        batch_size=batch_size,
    )
    _LOGGER.info("Opening form work store: %s", artifact_paths.database_path)
    with FormWorkStore(artifact_paths.database_path, state_key=state_key, resume=resume) as store:
        process_missing_form_batches(
            store,
            encoded_jsonl_path=encoded_jsonl_path,
            tokenization_config=tokenization_config,
            chord_decode=chord_decode,
            figure_min_n=figure_min_n,
            figure_max_n=figure_max_n,
            cadence_config=cadence_config,
            repetition_config=repetition_config,
            batch_size=batch_size,
            workers=workers,
            show_progress=show_progress,
        )
        _LOGGER.info("Exporting form statistics to %s", artifact_paths.root_directory)
        export_form_statistics(store, artifact_paths)


def export_form_statistics(store: FormWorkStore, artifact_paths: FormArtifactPaths) -> None:
    write_phrase_length_counts(store.tables.phrase_length_counts(), artifact_paths.phrase_lengths_path)
    write_segment_length_counts(store.tables.segment_length_counts(), artifact_paths.segment_lengths_path)
    write_closing_counts(store.tables.closing_counts(), artifact_paths.closings_path)
    write_histogram_counts(store.tables.similarity_histogram(), artifact_paths.similarity_histogram_path)
    write_histogram_counts(store.tables.best_match_histogram(), artifact_paths.best_match_histogram_path)
