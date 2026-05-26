import csv
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final, Protocol

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.artifacts import FIGURE_ALL_DIR_NAME, FigureArtifactPaths
from musak_model.n_grams.profile.io import (
    COUNT_COLUMN,
    FIGURE_COLUMN,
    HAND_COLUMN,
    N_COLUMN,
    SCALE_TYPE_COLUMN,
    read_figure_profile,
)
from musak_model.n_grams.profile.loading import (
    FigureProfileArtifacts,
    load_processed_figure_profile_artifacts,
)
from musak_model.n_grams.profile.metrics import figure_profile_comparison_metrics
from musak_model.n_grams.profile.schema import FigureProfile
from musak_model.n_grams.profile.streaming.executor import process_missing_sample_batches
from musak_model.n_grams.profile.streaming.export import export_figure_artifacts
from musak_model.n_grams.profile.streaming.store import (
    FigureWorkStore,
    complete_figure_artifacts_exist,
    figure_work_store_path,
)
from musak_model.paths import DEFAULT_TRAINING_FIGURE_DIR, N_GRAM_ANALYSIS_CONFIG_PATH
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit

_SPLIT_FIGURE_ARTIFACT_VERSION: Final[int] = 1
_LOGGER = logging.getLogger(__name__)


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...


@dataclass(frozen=True)
class SplitFigureArtifacts:
    profile: FigureProfile
    paths: FigureArtifactPaths


@dataclass(frozen=True)
class FigureCountGroup:
    key: tuple[str, str, int]
    counts: Counter[str]


def load_generation_figure_profile_artifacts(
    *,
    source_directory: Path,
    ingestion_config: IngestionConfig,
    tokenization_config: TokenizationConfig,
) -> FigureProfileArtifacts | None:
    if ingestion_config.processed_root is None:
        return None

    _LOGGER.info("Loading generation figure profile artifacts")
    started_at = perf_counter()
    artifacts = load_processed_figure_profile_artifacts(
        processed_root=ingestion_config.processed_root,
        dataset_root=source_directory,
        tokenization_config=tokenization_config,
    )
    if artifacts is None:
        _LOGGER.info("No generation figure profile artifacts found")
    else:
        _LOGGER.info(
            "Loaded generation figure profile artifacts in %.1fs: profile_groups=%s sample_profiles=%s",
            perf_counter() - started_at,
            len(artifacts.profile.groups),
            len(artifacts.sample_counts),
        )
    return artifacts


def split_figure_profile_metrics(
    split: IngestionSplit,
    *,
    token_vocabulary: TokenVocabulary,
    tokenization_config: TokenizationConfig,
    analysis_config_path: Path | None = None,
    artifact_root: Path = DEFAULT_TRAINING_FIGURE_DIR,
    workers: int,
    show_progress: bool = False,
) -> dict[str, float]:
    config_path = N_GRAM_ANALYSIS_CONFIG_PATH if analysis_config_path is None else analysis_config_path
    config = NGramAnalysisConfig.load(config_path).model_copy(update={"workers": max(1, workers)})
    split_key = _split_cache_key(
        split,
        config=config,
        token_vocabulary=token_vocabulary,
        tokenization_config=tokenization_config,
    )
    split_directory = artifact_root / split_key
    _LOGGER.info(
        "Computing train/validation figure metrics: train_samples=%s validation_samples=%s min_n=%s max_n=%s "
        "batch_size=%s workers=%s artifact_dir=%s",
        len(split.train),
        len(split.validation),
        config.min_n,
        config.max_n,
        config.batch_size,
        config.workers,
        split_directory,
    )
    train_artifacts = _split_artifacts(
        split.train,
        split_name="train",
        split_directory=split_directory,
        config=config,
        config_path=config_path,
        tokenization_config=tokenization_config,
        state_key=split_key,
        show_progress=show_progress,
    )
    validation_artifacts = _split_artifacts(
        split.validation,
        split_name="validation",
        split_directory=split_directory,
        config=config,
        config_path=config_path,
        tokenization_config=tokenization_config,
        state_key=split_key,
        show_progress=show_progress,
    )
    metrics = {
        **_split_figure_profile_count_metrics(
            train_profile=train_artifacts.profile,
            validation_profile=validation_artifacts.profile,
        ),
        **figure_profile_comparison_metrics(
            reference_profile=train_artifacts.profile,
            comparison_profile=validation_artifacts.profile,
            metric_prefix="model/split/figure",
            require_comparison_samples=True,
        ),
        **_figure_distribution_metrics_from_csv(
            reference_path=train_artifacts.paths.counts_path,
            comparison_path=validation_artifacts.paths.counts_path,
            metric_prefix="model/split/figure",
        ),
    }
    _LOGGER.info("Computed %s train/validation figure metric(s)", len(metrics))
    return metrics


def _split_figure_profile_count_metrics(
    *,
    train_profile: FigureProfile,
    validation_profile: FigureProfile,
) -> dict[str, float]:
    return {
        "model/split/figure/count/train_samples": float(train_profile.metadata.sample_count),
        "model/split/figure/count/validation_samples": float(validation_profile.metadata.sample_count),
        "model/split/figure/count/train_profile_groups": float(len(train_profile.groups)),
        "model/split/figure/count/validation_profile_groups": float(len(validation_profile.groups)),
    }


def _split_artifacts(
    samples: list[EncodedExercise],
    *,
    split_name: str,
    split_directory: Path,
    config: NGramAnalysisConfig,
    config_path: Path,
    tokenization_config: TokenizationConfig,
    state_key: str,
    show_progress: bool,
) -> SplitFigureArtifacts:
    paths = _split_artifact_paths(split_directory / split_name)
    task_count = (len(samples) + config.batch_size - 1) // config.batch_size
    if complete_figure_artifacts_exist(paths):
        _LOGGER.info("Reusing %s split figure artifacts: %s", split_name, paths.root_directory)
        return SplitFigureArtifacts(profile=read_figure_profile(paths.profile_path), paths=paths)

    _LOGGER.info(
        "Counting %s split figure n-grams: samples=%s batches=%s min_n=%s max_n=%s workers=%s artifact_dir=%s",
        split_name,
        len(samples),
        task_count,
        config.min_n,
        config.max_n,
        config.workers,
        paths.root_directory,
    )
    started_at = perf_counter()
    store_path = figure_work_store_path(paths)
    with FigureWorkStore(store_path, state_key=f"{state_key}:{split_name}", resume=True) as store:
        process_missing_sample_batches(
            store,
            samples=samples,
            tokenization_config=tokenization_config,
            config=config,
            show_progress=show_progress,
            progress_description=f"Counting {split_name} split figure n-gram batches",
        )
        export_figure_artifacts(
            store,
            artifact_paths=paths,
            output_path=None,
            analysis_config_path=config_path,
            min_n=config.min_n,
            max_n=config.max_n,
            limit_per_group=None,
        )

    store_path.unlink(missing_ok=True)
    _LOGGER.info(
        "Saved %s split figure artifacts in %.1fs: counts=%s profile=%s",
        split_name,
        perf_counter() - started_at,
        paths.counts_path,
        paths.profile_path,
    )
    return SplitFigureArtifacts(profile=read_figure_profile(paths.profile_path), paths=paths)


def _split_artifact_paths(root_directory: Path) -> FigureArtifactPaths:
    all_directory = root_directory / FIGURE_ALL_DIR_NAME
    return FigureArtifactPaths(
        root_directory=root_directory,
        config_path=root_directory / "config.yml",
        all_directory=all_directory,
        profile_path=all_directory / "profile.json",
        counts_path=all_directory / "counts.csv",
        by_sample_path=root_directory / "by_sample.jsonl",
    )


def _split_cache_key(
    split: IngestionSplit,
    *,
    config: NGramAnalysisConfig,
    token_vocabulary: TokenVocabulary,
    tokenization_config: TokenizationConfig,
) -> str:
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=token_vocabulary.duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "version": _SPLIT_FIGURE_ARTIFACT_VERSION,
                "tokenizer_hash": snapshot.tokenizer_hash,
                "min_n": config.min_n,
                "max_n": config.max_n,
                "batch_size": config.batch_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _update_samples_hash(hasher, "train", split.train)
    _update_samples_hash(hasher, "validation", split.validation)
    return hasher.hexdigest()


def _update_samples_hash(hasher: _Hasher, split_name: str, samples: list[EncodedExercise]) -> None:
    hasher.update(split_name.encode("utf-8"))
    hasher.update(str(len(samples)).encode("utf-8"))
    for sample in samples:
        hasher.update(sample.model_dump_json().encode("utf-8"))


def _figure_distribution_metrics_from_csv(
    *,
    reference_path: Path,
    comparison_path: Path,
    metric_prefix: str,
) -> dict[str, float]:
    distances: list[float] = []
    comparison_groups = _iter_count_groups(comparison_path)
    comparison_group = next(comparison_groups, None)
    for reference_group in _iter_count_groups(reference_path):
        if not reference_group.counts:
            continue

        while comparison_group is not None and comparison_group.key < reference_group.key:
            comparison_group = next(comparison_groups, None)

        if comparison_group is not None and comparison_group.key == reference_group.key:
            comparison_counts = comparison_group.counts
            comparison_group = next(comparison_groups, None)
        else:
            comparison_counts = Counter()

        distances.append(_total_variation_distance(reference_group.counts, comparison_counts))

    if not distances:
        return {f"{metric_prefix}/count/distribution_groups": 0.0}

    return {
        f"{metric_prefix}/count/distribution_groups": float(len(distances)),
        f"{metric_prefix}/mean/identity_total_variation_distance": sum(distances) / len(distances),
    }


def _iter_count_groups(path: Path) -> Iterator[FigureCountGroup]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        current_key: tuple[str, str, int] | None = None
        counts: Counter[str] = Counter()
        for row in reader:
            key = (row[SCALE_TYPE_COLUMN], row[HAND_COLUMN], int(row[N_COLUMN]))
            if current_key is not None and key != current_key:
                yield FigureCountGroup(key=current_key, counts=counts)
                counts = Counter()

            current_key = key
            counts[row[FIGURE_COLUMN]] += int(row[COUNT_COLUMN])

        if current_key is not None:
            yield FigureCountGroup(key=current_key, counts=counts)


def _total_variation_distance(reference_counts: Counter[str], comparison_counts: Counter[str]) -> float:
    reference_total = sum(reference_counts.values())
    if reference_total == 0:
        return 0.0

    comparison_total = sum(comparison_counts.values())
    figures = set(reference_counts) | set(comparison_counts)
    return 0.5 * sum(
        abs(
            (reference_counts[figure] / reference_total)
            - (comparison_counts[figure] / comparison_total if comparison_total > 0 else 0.0)
        )
        for figure in figures
    )
