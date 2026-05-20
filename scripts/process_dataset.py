import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from musak_model.data.config import (
    DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
    DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
    DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
    DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
    DataProcessingConfig,
    SegmentationMode,
    load_difficulty_labels,
    load_segmentation_config,
)
from musak_model.paths import (
    DEFAULT_PROCESSED_ROOT,
    SEGMENTATION_CONFIG_PATH,
    TOKENIZATION_CONFIG_PATH,
)
from musak_model.processing.dataset import process_dataset
from musak_model.processing.tracking import ProcessingMlflowConfig, build_processing_tracker
from musak_model.tokens.config import TokenizationConfig
from musak_shared.files import collect_musicxml_files

_LOGGER = logging.getLogger(__name__)


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
    tokenization_config = TokenizationConfig.load(args.tokenization_config)
    difficulty_labels = load_difficulty_labels(args.difficulty_labels)
    _LOGGER.info("Starting dataset processing")
    _LOGGER.info("Input directory: %s", args.data_dir)
    _LOGGER.info("Processed root: %s", args.processed_dir)
    _LOGGER.info("Resolved artifact directory: %s", args.processed_dir / args.data_dir.name)
    _LOGGER.info("Stage: %s", args.stage)
    _LOGGER.info("Workers: %s", args.workers)
    _LOGGER.info("Overwrite: %s", args.overwrite)
    _LOGGER.info("Progress bars: %s", not args.no_progress)
    _LOGGER.info("Segmentation config: %s", args.segmentation_config)
    _LOGGER.info("Segmentation mode: %s", segmentation_config.mode.value)
    _LOGGER.info("Tokenization config: %s", args.tokenization_config)
    _LOGGER.info("Remove segments with silent bars: %s", args.remove_segments_with_silent_bars)
    source_files = collect_musicxml_files(args.data_dir)
    if not source_files:
        _LOGGER.warning(
            "No MusicXML files found in %s. Expected files with .mxl, .xml, or .musicxml suffixes.",
            args.data_dir,
        )
        return

    _LOGGER.info("MusicXML files found: %s", len(source_files))
    tracker = build_processing_tracker(
        config=ProcessingMlflowConfig(
            enabled=not args.disable_mlflow,
            experiment_name=args.mlflow_experiment_name,
            run_name=args.mlflow_run_name,
            tracking_uri=args.mlflow_tracking_uri,
        )
    )
    with tracker:
        result = process_dataset(
            args.data_dir,
            processed_root=args.processed_dir,
            segmentation_config=segmentation_config,
            tokenization_config=tokenization_config,
            data_processing_config=DataProcessingConfig(
                remove_segments_with_silent_bars=args.remove_segments_with_silent_bars,
                scale_match_support_score_margin=args.scale_match_support_score_margin,
                scale_match_selection_score_margin=args.scale_match_selection_score_margin,
                scale_match_maximum_unexplained_weight_fraction=(args.scale_match_maximum_unexplained_weight_fraction),
                scale_match_maximum_explanation_pitch_class_count=(
                    args.scale_match_maximum_explanation_pitch_class_count
                ),
            ),
            stage=args.stage,
            difficulty_labels=difficulty_labels,
            overwrite=args.overwrite,
            workers=args.workers,
            show_progress=not args.no_progress,
        )
        tracker.log_processing_result(
            result=result,
            data_dir=args.data_dir,
            processed_root=args.processed_dir,
            stage=args.stage,
            overwrite=args.overwrite,
        )
    _LOGGER.info("Finished dataset processing")
    _LOGGER.info("Parsed manifest: %s", result.parsed_manifest_path)
    if result.encoded_manifest_path is not None:
        _LOGGER.info("Encoded manifest: %s", result.encoded_manifest_path)
    if result.tokenizer_snapshot_path is not None:
        _LOGGER.info("Tokenizer snapshot: %s", result.tokenizer_snapshot_path)
    _LOGGER.info(
        "Counts: parsed=%s encoded=%s errors=%s", result.parsed_count, result.encoded_count, result.error_count
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
            "  uv run python scripts/process_dataset.py --data-dir data/PDMX --stage parsed --no-progress\n\n"
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
        choices=("parsed", "encoded", "all"),
        default="all",
        help="Processing stage to run: parsed JSON only, encoded JSONL from parsed scores, or both.",
    )
    parser.add_argument(
        "--segmentation-config",
        type=Path,
        default=SEGMENTATION_CONFIG_PATH,
        help="YAML file containing default segmentation settings.",
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
        default=True,
        help="Mark any segment containing a fully silent bar as ineligible for training.",
    )
    parser.add_argument(
        "--scale-match-support-score-margin",
        type=float,
        default=DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
        help="Maximum score gap for alternate scale candidates that may explain chromatic pitch classes.",
    )
    parser.add_argument(
        "--scale-match-selection-score-margin",
        type=float,
        default=DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
        help="Maximum score gap for considering a more explanatory candidate over the strict best match.",
    )
    parser.add_argument(
        "--scale-match-maximum-unexplained-weight-fraction",
        type=float,
        default=DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
        help="Maximum duration-weighted pitch fraction not explained by the selected scale or close variants.",
    )
    parser.add_argument(
        "--scale-match-maximum-explanation-pitch-class-count",
        type=int,
        default=DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
        help="Maximum pitch-class count allowed in the selected scale plus close explanatory variants.",
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
        default=_default_worker_count(),
        help="Worker processes for MusicXML parsing. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum logging level.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parsed/encoded artifacts.")
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow dataset metric logging.")
    parser.add_argument(
        "--mlflow-experiment-name",
        default="musak-process",
        help="MLflow experiment name for processing metrics.",
    )
    parser.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name for processing metrics.")
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Optional MLflow tracking URI.")
    return parser.parse_args(argv)


def _default_worker_count() -> int:
    return max((os.cpu_count() or 2) - 1, 1)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    main()
