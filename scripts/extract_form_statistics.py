import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.harmony.decoding.config import ChordDecoderConfig
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.artifacts import figure_artifact_paths
from musak_model.n_grams.profile.chord.schema import ChordDecodeSpec
from musak_model.paths import DEFAULT_PROCESSED_ROOT, N_GRAM_ANALYSIS_CONFIG_PATH
from musak_model.processing.io import load_tokenizer_snapshot_json
from musak_model.processing.paths import ENCODED_JSONL_NAME, TOKENIZER_SNAPSHOT_NAME
from musak_model.synthetic.fitting.form.fit import FormFittingConfig
from musak_model.synthetic.fitting.form.io import FormArtifactPaths, form_artifact_paths_for_figure_root
from musak_model.synthetic.fitting.form.orchestration import extract_form_statistics
from musak_model.tokens.config import TokenizationConfig
from scripts.extract_figures import resolve_encoded_directory

_LOGGER = logging.getLogger(__name__)
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
_EXIT_FAILURE: Final[int] = 1


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    try:
        analysis_config = NGramAnalysisConfig.load(args.analysis_config)
        form_fitting = (
            FormFittingConfig.load()
            if args.form_fitting_config is None
            else FormFittingConfig.load(args.form_fitting_config)
        )
        encoded_directory = resolve_encoded_directory(
            data_directory=args.data_dir,
            processed_root=args.processed_root,
            encoded_directory=args.encoded_directory,
        )
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.error("Form statistics extraction input is invalid: %s", exception)
        raise SystemExit(_EXIT_FAILURE) from exception

    figure_root = figure_artifact_paths(encoded_directory).root_directory
    artifact_paths = form_artifact_paths_for_figure_root(figure_root)
    if args.overwrite:
        _clear_form_artifacts(artifact_paths)

    snapshot = load_tokenizer_snapshot_json(encoded_directory / TOKENIZER_SNAPSHOT_NAME)
    chord_decode = ChordDecodeSpec(decoder_config=ChordDecoderConfig.load(), vocabulary=ChordVocabularyConfig.load())
    _LOGGER.info("Resolved encoded directory: %s", encoded_directory)
    _LOGGER.info("Form artifact directory: %s", artifact_paths.root_directory)
    _LOGGER.info("Workers: %s, batch size: %s", analysis_config.execution.workers, analysis_config.execution.batch_size)
    try:
        extract_form_statistics(
            encoded_jsonl_path=encoded_directory / ENCODED_JSONL_NAME,
            artifact_paths=artifact_paths,
            tokenization_config=TokenizationConfig.model_validate(snapshot.tokenization_config),
            chord_decode=chord_decode,
            cadence_config=form_fitting.cadence,
            repetition_config=form_fitting.repetition,
            figure_min_n=analysis_config.figure_analysis.min_n,
            figure_max_n=analysis_config.figure_analysis.max_n,
            batch_size=analysis_config.execution.batch_size,
            workers=analysis_config.execution.workers,
            tokenizer_hash=snapshot.tokenizer_hash,
            show_progress=not args.no_progress,
            resume=args.resume,
        )
    except (FileNotFoundError, RuntimeError) as exception:
        _LOGGER.error("Form statistics extraction failed: %s", exception)
        raise SystemExit(_EXIT_FAILURE) from exception

    _LOGGER.info("Form statistics written under %s", artifact_paths.root_directory)


def _clear_form_artifacts(artifact_paths: FormArtifactPaths) -> None:
    for path in (
        artifact_paths.database_path,
        artifact_paths.phrase_lengths_path,
        artifact_paths.segment_lengths_path,
        artifact_paths.closings_path,
        artifact_paths.similarity_histogram_path,
        artifact_paths.best_match_histogram_path,
    ):
        path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-scale-type form statistics (phrase lengths, closing patterns, repetition "
        "similarity histograms) from an encoded dataset run into a resumable work store and parquet tables.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Dataset root. Resolves <processed-root>/<data-dir.name>/encoded/<run> as the encoded directory.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root directory for processed dataset artifacts.",
    )
    parser.add_argument(
        "--encoded-directory",
        type=Path,
        help=(
            f"Specific encoded run directory containing {ENCODED_JSONL_NAME} and {TOKENIZER_SNAPSHOT_NAME}. "
            "Required only when the processed dataset has multiple encoded runs."
        ),
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=N_GRAM_ANALYSIS_CONFIG_PATH,
        help="N-gram analysis YAML config (supplies figure n range, batch size and worker count).",
    )
    parser.add_argument(
        "--form-fitting-config",
        type=Path,
        help="Form-fitting YAML config supplying the cadence detector and repetition settings for the corpus pass.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum logging level.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing form statistics before extraction.")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete form statistics extraction.")
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=_LOG_LEVELS[level], format="%(asctime)s %(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    main()
