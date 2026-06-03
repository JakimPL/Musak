import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.paths import (
    DEFAULT_DIAGNOSTIC_DIRECTORY,
    DEFAULT_MLFLOW_DB_PATH,
    DEFAULT_PROCESSED_ROOT,
    PROCESSING_CONFIG_PATH,
)
from musak_model.processing.config import ProcessingConfig
from musak_model.processing.diagnostic_report import DEFAULT_TOP_ROWS, write_dataset_diagnostic_report
from musak_model.processing.encoded_runs import resolve_encoded_directory
from scripts.utils.logger import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging

_LOGGER = logging.getLogger(__name__)
_EXIT_FAILURE: Final[int] = 1
_DEFAULT_MAX_SEQUENCE_LENGTH: Final[int] = 1024


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)
    try:
        encoded_directory = resolve_encoded_directory(
            data_directory=args.data_dir,
            processed_root=args.processed_root,
            encoded_directory=args.encoded_directory,
        )
        reference_encoded_directory = _resolve_reference_encoded_directory(args)
        output_directory = _output_directory(
            configured=args.output_dir,
            data_directory=args.data_dir,
            encoded_directory=encoded_directory,
        )
        processing_config = ProcessingConfig.load(args.processing_config)
        result = write_dataset_diagnostic_report(
            dataset_name=dataset_name(data_directory=args.data_dir, encoded_directory=encoded_directory),
            processed_directory=processed_directory(
                data_directory=args.data_dir,
                processed_root=args.processed_root,
                encoded_directory=encoded_directory,
            ),
            encoded_directory=encoded_directory,
            output_directory=output_directory,
            scale_matcher_config=processing_config.tokenization.scale_matcher,
            reference_dataset_name=(
                dataset_name(data_directory=args.reference_data_dir, encoded_directory=reference_encoded_directory)
                if reference_encoded_directory is not None
                else None
            ),
            reference_processed_directory=(
                processed_directory(
                    data_directory=args.reference_data_dir,
                    processed_root=args.processed_root,
                    encoded_directory=reference_encoded_directory,
                )
                if reference_encoded_directory is not None
                else None
            ),
            reference_encoded_directory=reference_encoded_directory,
            max_sequence_length=args.max_sequence_length,
            top_rows=args.top_rows,
            mlflow_db_path=None if args.disable_mlflow_lookup else args.mlflow_db,
        )
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.error("Dataset diagnostics input is invalid: %s", exception)
        raise SystemExit(_EXIT_FAILURE) from exception

    _LOGGER.info("Diagnostic report written to %s", result.report_path)
    _LOGGER.info("Diagnostic summary written to %s", result.summary_path)
    _LOGGER.info("Diagnostic tables written under %s", result.output_directory / "tables")


def dataset_name(*, data_directory: Path | None, encoded_directory: Path | None) -> str:
    if data_directory is not None:
        return data_directory.name

    if encoded_directory is None:
        raise ValueError("encoded_directory is required when data_directory is omitted")

    return encoded_directory.parent.parent.name


def processed_directory(*, data_directory: Path | None, processed_root: Path, encoded_directory: Path | None) -> Path:
    if encoded_directory is not None:
        return encoded_directory.parent.parent

    if data_directory is not None:
        return processed_root / data_directory.name

    raise ValueError("data_directory or encoded_directory is required")


def _resolve_reference_encoded_directory(args: argparse.Namespace) -> Path | None:
    if args.reference_data_dir is None and args.reference_encoded_directory is None:
        return None

    return resolve_encoded_directory(
        data_directory=args.reference_data_dir,
        processed_root=args.processed_root,
        encoded_directory=args.reference_encoded_directory,
    )


def _output_directory(
    *,
    configured: Path | None,
    data_directory: Path | None,
    encoded_directory: Path,
) -> Path:
    if configured is not None:
        return configured

    name = dataset_name(data_directory=data_directory, encoded_directory=encoded_directory)
    return DEFAULT_DIAGNOSTIC_DIRECTORY / name / encoded_directory.name


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic diagnostic report from processed dataset artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Dataset root. Used to resolve <processed-root>/<data-dir.name>/encoded.",
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
        help="Specific encoded run directory. Required only when the processed dataset has multiple encoded runs.",
    )
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        help="Optional reference dataset root for distribution comparison.",
    )
    parser.add_argument(
        "--reference-encoded-directory",
        type=Path,
        help="Specific reference encoded run directory.",
    )
    parser.add_argument(
        "--processing-config",
        type=Path,
        default=PROCESSING_CONFIG_PATH,
        help="Processing YAML whose scale-matcher config is used for fixed tonal probes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report output directory. Defaults to artifacts/diagnostics/<dataset>/<tokenizer-hash>.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=_DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Model sequence limit used to count overlength encoded samples.",
    )
    parser.add_argument(
        "--top-rows",
        type=int,
        default=DEFAULT_TOP_ROWS,
        help="Number of outlier rows to keep per outlier table.",
    )
    parser.add_argument(
        "--mlflow-db",
        type=Path,
        default=DEFAULT_MLFLOW_DB_PATH,
        help="Local MLflow SQLite database for optional run-history lookup.",
    )
    parser.add_argument("--disable-mlflow-lookup", action="store_true", help="Skip MLflow database lookup.")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=DEFAULT_LOG_LEVEL,
        help="Minimum logging level.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
