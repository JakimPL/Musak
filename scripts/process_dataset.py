import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.analysis.n_grams import FigureExtractionResult, extract_figure_artifacts
from musak_model.data.config import (
    SegmentationConfig,
    SegmentationMode,
    load_difficulty_labels,
    load_segmentation_config,
)
from musak_model.paths import (
    DEFAULT_COMBINED_PROCESSING_PROFILE_OUTPUT_DIR,
    DEFAULT_PARSING_PROFILE_OUTPUT_DIR,
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR,
    DEFAULT_TOKENIZATION_PROFILE_OUTPUT_DIR,
    N_GRAM_ANALYSIS_CONFIG_PATH,
    PROCESSING_CONFIG_PATH,
    SEGMENTATION_CONFIG_PATH,
    TOKENIZATION_CONFIG_PATH,
)
from musak_model.processing.config import ProcessingConfig, processing_config_with_overrides
from musak_model.processing.dataset import ProcessDatasetResult, process_dataset
from musak_model.processing.profiler import build_processing_profiler
from musak_model.processing.tracking import ProcessingMlflowConfig, build_processing_tracker
from musak_model.tokens.config import TokenizationConfig
from musak_shared.files import collect_musicxml_files

_LOGGER = logging.getLogger(__name__)
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
_EXIT_FAILURE: Final[int] = 1
_PARSING_STAGES: Final[frozenset[str]] = frozenset({"parse", "process"})
_DEFAULT_PROFILE_OUTPUT_DIRS: Final[dict[str, Path]] = {
    "parse": DEFAULT_PARSING_PROFILE_OUTPUT_DIR,
    "tokenize": DEFAULT_TOKENIZATION_PROFILE_OUTPUT_DIR,
    "process": DEFAULT_COMBINED_PROCESSING_PROFILE_OUTPUT_DIR,
}


class _ProcessDatasetHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    segmentation_config = load_segmentation_config(
        args.segmentation_config,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        mode=SegmentationMode.WHOLE_FILE if args.whole_file_segments else None,
    )
    processing_config = processing_config_with_overrides(
        ProcessingConfig.load(args.processing_config),
        workers=args.workers,
        tokenization_workers=args.tokenization_workers,
        tokenization_batch_size=args.tokenization_batch_size,
        remove_segments_with_silent_bars=args.remove_segments_with_silent_bars,
        scale_match_support_score_margin=args.scale_match_support_score_margin,
        scale_match_selection_score_margin=args.scale_match_selection_score_margin,
        scale_match_maximum_unexplained_weight_fraction=args.scale_match_maximum_unexplained_weight_fraction,
        scale_match_maximum_explanation_pitch_class_count=args.scale_match_maximum_explanation_pitch_class_count,
    )
    tokenization_config = TokenizationConfig.load(args.tokenization_config)
    difficulty_labels = load_difficulty_labels(args.difficulty_labels)
    profile_output_dir = _profile_output_dir(args.stage, configured=args.profile_output_dir)
    source_files = _source_files_for_stage(args.data_dir, stage=args.stage)
    _log_processing_start(
        args,
        segmentation_config=segmentation_config,
        processing_config=processing_config,
        profile_output_dir=profile_output_dir,
        source_files=source_files,
    )
    profiler = build_processing_profiler(enabled=args.profile, output_dir=profile_output_dir)
    tracker = build_processing_tracker(
        config=ProcessingMlflowConfig(
            enabled=not args.disable_mlflow and not args.profile,
            experiment_name=args.mlflow_experiment_name,
            run_name=args.mlflow_run_name,
            tracking_uri=args.mlflow_tracking_uri,
        )
    )
    try:
        with tracker, profiler:
            result = process_dataset(
                args.data_dir,
                processed_root=args.processed_dir,
                segmentation_config=segmentation_config,
                tokenization_config=tokenization_config,
                processing_config=processing_config,
                stage=args.stage,
                difficulty_labels=difficulty_labels,
                overwrite=args.overwrite,
                show_progress=not args.no_progress,
                profiler=profiler,
            )
            profiler.step()
            tracker.log_processing_result(
                result=result,
                data_dir=args.data_dir,
                processed_root=args.processed_dir,
                stage=args.stage,
                overwrite=args.overwrite,
            )
            figure_result = extract_process_figure_artifacts(
                result=result,
                analysis_config_path=args.analysis_config,
                output_path=args.analysis_output,
                skip_figure_analysis=args.skip_figure_analysis,
                show_progress=not args.no_progress,
            )
            if figure_result is not None:
                tracker.log_figure_extraction_result(figure_result)
    except FileNotFoundError as exception:
        _log_processing_file_not_found(exception, data_dir=args.data_dir, processed_dir=args.processed_dir)
        raise SystemExit(_EXIT_FAILURE) from exception
    if profiler.enabled:
        profiler.write_reports()
        _LOGGER.info("Profile reports written to %s", profile_output_dir)
    _LOGGER.info("Finished dataset processing")
    _LOGGER.info("Parsed manifest: %s", result.parsed_manifest_path)
    if result.encoded_manifest_path is not None:
        _LOGGER.info("Encoded manifest: %s", result.encoded_manifest_path)
    if result.tokenizer_snapshot_path is not None:
        _LOGGER.info("Tokenizer snapshot: %s", result.tokenizer_snapshot_path)
    _LOGGER.info(
        "Counts: parsed=%s encoded=%s errors=%s", result.parsed_count, result.encoded_count, result.error_count
    )


def _log_processing_start(
    args: argparse.Namespace,
    *,
    segmentation_config: SegmentationConfig,
    processing_config: ProcessingConfig,
    profile_output_dir: Path,
    source_files: list[Path],
) -> None:
    _LOGGER.info("Starting dataset processing")
    _LOGGER.info("Input directory: %s", args.data_dir)
    _LOGGER.info("Processed root: %s", args.processed_dir)
    _LOGGER.info("Resolved artifact directory: %s", args.processed_dir / args.data_dir.name)
    _LOGGER.info("Stage: %s", args.stage)
    _LOGGER.info("Processing config: %s", args.processing_config)
    _LOGGER.info("Parsing workers: %s", processing_config.parsing.workers)
    _LOGGER.info("Tokenization workers: %s", processing_config.tokenization.workers)
    _LOGGER.info("Tokenization batch size: %s", processing_config.tokenization.batch_size)
    _LOGGER.info("Overwrite: %s", args.overwrite)
    _LOGGER.info("Progress bars: %s", not args.no_progress)
    _LOGGER.info("Segmentation config: %s", args.segmentation_config)
    _LOGGER.info("Segmentation mode: %s", segmentation_config.mode.value)
    _LOGGER.info("Tokenization config: %s", args.tokenization_config)
    _LOGGER.info("Figure analysis config: %s", args.analysis_config)
    _LOGGER.info("Figure analysis output: %s", args.analysis_output or "none")
    _LOGGER.info("Figure analysis enabled: %s", not args.skip_figure_analysis)
    _LOGGER.info(
        "Remove segments with silent bars: %s",
        processing_config.tokenization.remove_segments_with_silent_bars,
    )
    _LOGGER.info("Profiler enabled: %s", args.profile)
    if args.profile:
        _LOGGER.info("Profile output directory: %s", profile_output_dir)
    if source_files:
        _LOGGER.info("MusicXML files found: %s", len(source_files))


def extract_process_figure_artifacts(
    *,
    result: ProcessDatasetResult,
    analysis_config_path: Path,
    output_path: Path | None,
    skip_figure_analysis: bool,
    show_progress: bool,
) -> FigureExtractionResult | None:
    if skip_figure_analysis or result.encoded_manifest_path is None:
        return None

    encoded_dir = result.encoded_manifest_path.parent
    _LOGGER.info("Starting figure n-gram extraction for %s", encoded_dir)
    figure_result = extract_figure_artifacts(
        encoded_dir=encoded_dir,
        analysis_config_path=analysis_config_path,
        output_path=output_path,
        show_progress=show_progress,
    )
    _LOGGER.info("Figure profile groups: %s", figure_result.profile_group_count)
    _LOGGER.info("Figure n-gram counts written to %s", figure_result.artifact_paths.counts_path)
    _LOGGER.info("Figure profile written to %s", figure_result.artifact_paths.profile_path)
    if figure_result.extra_output_path is not None:
        _LOGGER.info("Extra figure n-gram counts written to %s", figure_result.extra_output_path)

    return figure_result


def _source_files_for_stage(data_dir: Path, *, stage: str) -> list[Path]:
    if stage not in _PARSING_STAGES:
        return []

    source_files = collect_musicxml_files(data_dir)
    if not source_files:
        _LOGGER.error(
            "No MusicXML files found in %s. Expected files with .mxl, .xml, or .musicxml suffixes. "
            "Processing cannot continue because no parsed manifest was created.",
            data_dir,
        )
        raise SystemExit(_EXIT_FAILURE)

    return source_files


def _profile_output_dir(stage: str, *, configured: Path | None) -> Path:
    if configured is not None:
        return configured

    return _DEFAULT_PROFILE_OUTPUT_DIRS[stage]


def _log_processing_file_not_found(
    exception: FileNotFoundError,
    *,
    data_dir: Path,
    processed_dir: Path,
) -> None:
    _LOGGER.error("Dataset processing input is missing: %s", exception)
    _LOGGER.error(
        "If you are tokenizing, run the parse stage first with the same --data-dir and --processed-dir, "
        "or run --stage process. Current artifact directory is %s.",
        processed_dir / data_dir.name,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process a MusicXML dataset into reusable parsed-score and encoded-segment artifacts. "
            "The input directory is searched recursively for .mxl, .xml, and .musicxml files."
        ),
        epilog=(
            "Examples:\n"
            "  uv run python scripts/process_dataset.py --data-dir data/PDMX\n"
            "  uv run python scripts/process_dataset.py --data-dir data/PDMX --processed-dir processed --workers 8\n"
            "  uv run python scripts/process_dataset.py --data-dir data/PDMX --stage parse --no-progress\n\n"
            "Output layout:\n"
            "  Artifacts are written below <processed-dir>/<data-dir.name>/.\n"
            "  For example, --data-dir data/PDMX --processed-dir processed writes to processed/PDMX/.\n"
            "  Pass the dataset root such as data/PDMX, not an internal folder such as data/PDMX/mxl."
        ),
        formatter_class=_ProcessDatasetHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root to search recursively for MusicXML files. The output namespace is this directory name.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root directory for generated intermediate artifacts.",
    )
    parser.add_argument(
        "--stage",
        choices=("parse", "tokenize", "process"),
        default="process",
        help="Processing stage to run: parsed JSON only, encoded JSONL from parsed scores, or both.",
    )
    parser.add_argument(
        "--segmentation-config",
        type=Path,
        default=SEGMENTATION_CONFIG_PATH,
        help="YAML file containing default segmentation settings.",
    )
    parser.add_argument(
        "--processing-config",
        type=Path,
        default=PROCESSING_CONFIG_PATH,
        help="YAML file containing parsing and tokenization processing settings.",
    )
    parser.add_argument(
        "--tokenization-config",
        type=Path,
        default=TOKENIZATION_CONFIG_PATH,
        help="YAML file containing tokenization and duration-vocabulary settings.",
    )
    parser.add_argument("--window-bars", type=int, default=None, help="Override segmentation window size in bars.")
    parser.add_argument("--stride-bars", type=int, default=None, help="Override segmentation stride in bars.")
    parser.add_argument(
        "--whole-file-segments",
        action="store_true",
        help="Encode each source file as one segment and populate bar_count with the full file length.",
    )
    parser.add_argument(
        "--remove-segments-with-silent-bars",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override processing config: mark any segment containing a fully silent bar as ineligible for training.",
    )
    parser.add_argument(
        "--scale-match-support-score-margin",
        type=float,
        default=None,
        help="Override processing config: maximum score gap for alternate scale candidates.",
    )
    parser.add_argument(
        "--scale-match-selection-score-margin",
        type=float,
        default=None,
        help="Override processing config: maximum score gap for considering a more explanatory candidate.",
    )
    parser.add_argument(
        "--scale-match-maximum-unexplained-weight-fraction",
        type=float,
        default=None,
        help="Override processing config: maximum unexplained duration-weighted pitch fraction.",
    )
    parser.add_argument(
        "--scale-match-maximum-explanation-pitch-class-count",
        type=int,
        default=None,
        help="Override processing config: maximum pitch-class count in selected scale plus close variants.",
    )
    parser.add_argument(
        "--difficulty-labels",
        type=Path,
        default=None,
        help="Optional YAML mapping from file stem to integer difficulty label.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Override processing config: worker processes for MusicXML parsing. "
            "Use 1 to disable parse multiprocessing."
        ),
    )
    parser.add_argument(
        "--tokenization-workers",
        type=int,
        default=None,
        help="Override processing config: worker processes for tokenization. Use 1 for serial tokenization.",
    )
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=None,
        help="Override processing config: parsed source files per tokenization worker task.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum logging level.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parsed/encoded artifacts.")
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=N_GRAM_ANALYSIS_CONFIG_PATH,
        help="YAML file containing figure n-gram analysis settings.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help="Optional extra figure n-gram CSV output path.",
    )
    parser.add_argument(
        "--skip-figure-analysis",
        action="store_true",
        help="Skip figure n-gram/profile extraction after tokenization.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Write processing timing and torch profiler reports without logging the run to MLflow.",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for processing timing and profiler reports. "
            f"Defaults to stage-specific directories under {DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR}."
        ),
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow dataset metric logging.")
    parser.add_argument(
        "--mlflow-experiment-name",
        default="musak-process",
        help="MLflow experiment name for processing metrics.",
    )
    parser.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name for processing metrics.")
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Optional MLflow tracking URI.")
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=_LOG_LEVELS[level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    main()
