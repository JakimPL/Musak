import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.analysis.n_grams.config import NGramAnalysisConfig
from musak_model.analysis.n_grams.profile.artifacts import figure_artifact_paths
from musak_model.analysis.n_grams.profile.extraction import extract_figure_artifacts
from musak_model.paths import DEFAULT_PROCESSED_ROOT, N_GRAM_ANALYSIS_CONFIG_PATH
from musak_model.processing.paths import ENCODED_JSONL_NAME, TOKENIZER_SNAPSHOT_NAME

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
    _LOGGER.info("Configured encoded directory: %s", args.encoded_directory)
    _LOGGER.info("Analysis config: %s", args.analysis_config)
    _LOGGER.info("Progress bars: %s", not args.no_progress)
    try:
        config = NGramAnalysisConfig.load(args.analysis_config)
        encoded_directory = resolve_encoded_directory(
            data_directory=args.data_dir,
            processed_root=args.processed_root,
            encoded_directory=args.encoded_directory,
        )
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.error("Figure n-gram extraction input is invalid: %s", exception)
        raise SystemExit(_EXIT_FAILURE) from exception

    artifact_paths = figure_artifact_paths(encoded_directory)
    output_path = default_output_path(data_dir=args.data_dir, encoded_directory=encoded_directory)
    encoded_jsonl_path = encoded_directory / ENCODED_JSONL_NAME
    tokenizer_snapshot_path = encoded_directory / TOKENIZER_SNAPSHOT_NAME
    _LOGGER.info("Resolved encoded directory: %s", encoded_directory)
    _LOGGER.info("Encoded JSONL: %s", encoded_jsonl_path)
    _LOGGER.info("Tokenizer snapshot: %s", tokenizer_snapshot_path)
    _LOGGER.info("Canonical count output: %s", output_path)
    _LOGGER.info("Canonical profile output: %s", artifact_paths.profile_path)
    _LOGGER.info("Extra CSV output: %s", args.output or "none")
    _LOGGER.info("n range: %s..%s", config.min_n, config.max_n)
    _LOGGER.info("Limit per group: %s", config.limit_per_group)
    _LOGGER.info("Workers: %s", config.workers)
    _LOGGER.info("Batch size: %s", config.batch_size)
    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
        analysis_config_path=args.analysis_config,
        output_path=args.output,
        show_progress=not args.no_progress,
    )
    _LOGGER.info("Encoded samples loaded: %s", result.encoded_sample_count)
    _LOGGER.info("Figure profile groups: %s", result.profile_group_count)
    _LOGGER.info("Figure sample profiles: %s", result.sample_profile_count)
    _LOGGER.info("Figure n-gram counts written to %s", output_path)
    _LOGGER.info("Figure profile written to %s", artifact_paths.profile_path)
    _LOGGER.info("Figure sample profiles written to %s", artifact_paths.by_sample_path)
    if args.output is not None:
        _LOGGER.info("Extra figure n-gram counts written to %s", args.output)

    _LOGGER.info("Finished figure n-gram extraction")


def default_output_path(
    *,
    data_dir: Path | None,
    encoded_directory: Path,
) -> Path:
    _ = data_dir
    return figure_artifact_paths(encoded_directory).counts_path


def dataset_name_for_analysis(
    *,
    data_directory: Path | None,
    encoded_directory: Path,
) -> str | None:
    if data_directory is not None:
        return data_directory.name

    if encoded_directory.parent.name == "encoded":
        return encoded_directory.parent.parent.name

    return None


def resolve_encoded_directory(
    *,
    data_directory: Path | None,
    processed_root: Path,
    encoded_directory: Path | None,
) -> Path:
    if encoded_directory is not None:
        return encoded_directory

    if data_directory is None:
        raise ValueError("--data-dir is required when --encoded-dir is omitted")

    encoded_root = processed_root / data_directory.name / "encoded"
    encoded_directorys = encoded_run_directories(encoded_root)
    if not encoded_directorys:
        raise FileNotFoundError(f"No encoded runs found in {encoded_root}")

    if len(encoded_directorys) > 1:
        raise ValueError(f"Multiple encoded runs found in {encoded_root}; pass --encoded-dir explicitly")

    return encoded_directorys[0]


def encoded_run_directories(encoded_root: Path) -> list[Path]:
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
        help="Extra CSV output path. Canonical counts are always written under <encoded-dir>/figure/all/counts.csv.",
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
