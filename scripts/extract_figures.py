import argparse
import csv
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.analysis.n_grams import (
    COUNT_CSV_COLUMNS,
    NGramAnalysisConfig,
    count_encoded_exercises_figure_ngrams,
    figure_count_records,
    write_figure_count_csv,
)
from musak_model.paths import DEFAULT_ANALYSIS_DIR, DEFAULT_PROCESSED_ROOT, N_GRAM_ANALYSIS_CONFIG_PATH
from musak_model.processing.io import load_encoded_jsonl, load_tokenizer_snapshot_json
from musak_model.processing.paths import ENCODED_JSONL_NAME, TOKENIZER_SNAPSHOT_NAME
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary

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
    _LOGGER.info("Starting figure n-gram extraction")
    _LOGGER.info("Data directory: %s", args.data_dir)
    _LOGGER.info("Processed root: %s", args.processed_root)
    _LOGGER.info("Configured encoded directory: %s", args.encoded_dir)
    _LOGGER.info("Analysis config: %s", args.analysis_config)
    _LOGGER.info("Progress bars: %s", not args.no_progress)
    try:
        config = NGramAnalysisConfig.load(args.analysis_config)
        encoded_dir = resolve_encoded_dir(
            data_dir=args.data_dir,
            processed_root=args.processed_root,
            encoded_dir=args.encoded_dir,
        )
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.error("Figure n-gram extraction input is invalid: %s", exception)
        raise SystemExit(_EXIT_FAILURE) from exception

    output_path = args.output or default_output_path(data_dir=args.data_dir, encoded_dir=encoded_dir)
    encoded_jsonl_path = encoded_dir / ENCODED_JSONL_NAME
    tokenizer_snapshot_path = encoded_dir / TOKENIZER_SNAPSHOT_NAME
    _LOGGER.info("Resolved encoded directory: %s", encoded_dir)
    _LOGGER.info("Encoded JSONL: %s", encoded_jsonl_path)
    _LOGGER.info("Tokenizer snapshot: %s", tokenizer_snapshot_path)
    _LOGGER.info("Output path: %s", output_path or "stdout")
    _LOGGER.info("n range: %s..%s", config.min_n, config.max_n)
    _LOGGER.info("Limit per group: %s", config.limit_per_group)
    snapshot = load_tokenizer_snapshot_json(tokenizer_snapshot_path)
    tokenization_config = TokenizationConfig.model_validate(snapshot.tokenization_config)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    samples = load_encoded_jsonl(encoded_jsonl_path)
    _LOGGER.info("Encoded samples loaded: %s", len(samples))
    counts = count_encoded_exercises_figure_ngrams(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=config.min_n,
        max_n=config.max_n,
        show_progress=not args.no_progress,
    )
    records = figure_count_records(counts, limit_per_group=config.limit_per_group)
    _LOGGER.info("Figure count records: %s", len(records))
    if output_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=COUNT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
        _LOGGER.info("Finished figure n-gram extraction")
        return

    write_figure_count_csv(records, output_path)
    _LOGGER.info("Figure n-gram counts written to %s", output_path)
    _LOGGER.info("Finished figure n-gram extraction")


def default_output_path(
    *,
    data_dir: Path | None,
    encoded_dir: Path,
) -> Path | None:
    dataset_name = dataset_name_for_analysis(data_dir=data_dir, encoded_dir=encoded_dir)
    if dataset_name is None:
        return None

    return DEFAULT_ANALYSIS_DIR / f"{dataset_name}.csv"


def dataset_name_for_analysis(
    *,
    data_dir: Path | None,
    encoded_dir: Path,
) -> str | None:
    if data_dir is not None:
        return data_dir.name

    if encoded_dir.parent.name == "encoded":
        return encoded_dir.parent.parent.name

    return None


def resolve_encoded_dir(
    *,
    data_dir: Path | None,
    processed_root: Path,
    encoded_dir: Path | None,
) -> Path:
    if encoded_dir is not None:
        return encoded_dir

    if data_dir is None:
        raise ValueError("--data-dir is required when --encoded-dir is omitted")

    encoded_root = processed_root / data_dir.name / "encoded"
    encoded_dirs = encoded_run_dirs(encoded_root)
    if not encoded_dirs:
        raise FileNotFoundError(f"No encoded runs found in {encoded_root}")

    if len(encoded_dirs) > 1:
        raise ValueError(f"Multiple encoded runs found in {encoded_root}; pass --encoded-dir explicitly")

    return encoded_dirs[0]


def encoded_run_dirs(encoded_root: Path) -> list[Path]:
    if not encoded_root.exists():
        return []

    return sorted(
        path
        for path in encoded_root.iterdir()
        if path.is_dir() and (path / ENCODED_JSONL_NAME).is_file() and (path / TOKENIZER_SNAPSHOT_NAME).is_file()
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract scale-grouped, per-hand figure n-gram counts from an encoded dataset run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Dataset root. Used to resolve <processed-root>/<data-dir.name>/encoded and default output name.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root directory for processed dataset artifacts.",
    )
    parser.add_argument(
        "--encoded-dir",
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
        help="N-gram analysis YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"CSV output path. Defaults to {DEFAULT_ANALYSIS_DIR}/<dataset-name>.csv when a dataset name is known.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum logging level.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=_LOG_LEVELS[level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    main()
