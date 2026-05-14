import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from musak_model.common.files import collect_musicxml_files
from musak_model.data.config import load_difficulty_labels, load_segmentation_config
from musak_model.paths import (
    DEFAULT_PROCESSED_ROOT,
    SEGMENTATION_CONFIG_PATH,
    TOKENIZATION_CONFIG_PATH,
)
from musak_model.processing.dataset import process_dataset
from musak_model.tokens.config import TokenizationConfig

_LOGGER = logging.getLogger(__name__)


class _ProcessDatasetHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    segmentation = load_segmentation_config(
        args.segmentation_config,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
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
    _LOGGER.info("Tokenization config: %s", args.tokenization_config)
    source_files = collect_musicxml_files(args.data_dir)
    if not source_files:
        _LOGGER.warning(
            "No MusicXML files found in %s. Expected files with .mxl, .xml, or .musicxml suffixes.",
            args.data_dir,
        )
        return

    _LOGGER.info("MusicXML files found: %s", len(source_files))
    result = process_dataset(
        args.data_dir,
        processed_root=args.processed_dir,
        segmentation=segmentation,
        tokenization_config=tokenization_config,
        stage=args.stage,
        difficulty_labels=difficulty_labels,
        overwrite=args.overwrite,
        workers=args.workers,
        show_progress=not args.no_progress,
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
