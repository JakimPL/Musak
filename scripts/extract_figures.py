import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

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


def main() -> None:
    args = _parse_args()
    config = NGramAnalysisConfig.load(args.analysis_config)
    encoded_dir = resolve_encoded_dir(
        data_dir=args.data_dir,
        processed_root=args.processed_root,
        encoded_dir=args.encoded_dir,
    )
    output_path = args.output or default_output_path(data_dir=args.data_dir, encoded_dir=encoded_dir)
    encoded_jsonl_path = encoded_dir / ENCODED_JSONL_NAME
    tokenizer_snapshot_path = encoded_dir / TOKENIZER_SNAPSHOT_NAME
    snapshot = load_tokenizer_snapshot_json(tokenizer_snapshot_path)
    tokenization_config = TokenizationConfig.model_validate(snapshot.tokenization_config)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    samples = load_encoded_jsonl(encoded_jsonl_path)
    counts = count_encoded_exercises_figure_ngrams(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=config.min_n,
        max_n=config.max_n,
    )
    records = figure_count_records(counts, limit_per_group=config.limit_per_group)
    if output_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=COUNT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
        return

    write_figure_count_csv(records, output_path)


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
