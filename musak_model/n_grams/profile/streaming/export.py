import csv
import shutil
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

from musak_model.n_grams.figure.signature import figure_signature_from_json, figure_signature_to_ngram
from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.io import (
    BASE_DURATION_COLUMN,
    BASE_DURATION_CSV_COLUMNS,
    COUNT_COLUMN,
    COUNT_CSV_COLUMNS,
    FIGURE_COLUMN,
    HAND_COLUMN,
    N_COLUMN,
    SCALE_TYPE_COLUMN,
)
from musak_model.n_grams.profile.rhythm.io import build_rhythm_profile, write_rhythm_counts_csv, write_rhythm_profile
from musak_model.n_grams.profile.rhythm.schema import (
    RhythmCountCounter,
    RhythmProfileMetadata,
    rhythm_artifact_paths_for_figure_root,
)
from musak_model.n_grams.profile.schema import FigureProfile, FigureProfileGroup, FigureProfileMetadata
from musak_model.n_grams.profile.streaming.schema import FigureCountKey, FigureStoreSummary
from musak_model.n_grams.profile.streaming.store import FigureWorkStore
from musak_model.n_grams.profile.streaming.totals import figure_group_totals
from musak_model.processing.io import JSON_INDENT
from musak_model.tokens.schema import Hand, ScaleType


def export_figure_artifacts(
    store: FigureWorkStore,
    *,
    artifact_paths: FigureArtifactPaths,
    output_path: Path | None,
    analysis_config_path: Path,
    min_n: int,
    max_n: int,
    rhythm_min_n: int,
    rhythm_max_n: int,
    grid_alignment_denominators: tuple[int, ...],
    strong_beat_offsets: tuple[Fraction, ...],
    limit_per_group: int | None,
) -> FigureStoreSummary:
    profile = profile_from_store(store, min_n=min_n, max_n=max_n)
    rhythm_counts = rhythm_counts_from_store(store)
    rhythm_paths = rhythm_artifact_paths_for_figure_root(artifact_paths.root_directory)
    rhythm_profile = build_rhythm_profile(
        rhythm_counts,
        metadata=RhythmProfileMetadata(
            rhythm_min_n=rhythm_min_n,
            rhythm_max_n=rhythm_max_n,
            grid_alignment_denominators=grid_alignment_denominators,
            strong_beat_offsets=strong_beat_offsets,
            sample_count=store.encoded_sample_count(),
        ),
    )
    sample_profile_count = export_sample_counts(store, artifact_paths.by_sample_path)
    export_counts_csv(store, artifact_paths.counts_path, limit_per_group=None)
    export_base_durations_csv(store, artifact_paths.base_durations_path)
    write_rhythm_counts_csv(rhythm_counts, rhythm_paths.counts_path)
    write_rhythm_profile(rhythm_profile, rhythm_paths.profile_path)
    write_profile_atomically(profile, artifact_paths.profile_path)
    copy_file_atomically(analysis_config_path, artifact_paths.config_path)
    if output_path is not None:
        export_counts_csv(store, output_path, limit_per_group=limit_per_group)

    return FigureStoreSummary(
        encoded_sample_count=store.encoded_sample_count(),
        profile_group_count=len(profile.groups),
        sample_profile_count=sample_profile_count,
    )


def rhythm_counts_from_store(store: FigureWorkStore) -> RhythmCountCounter:
    return store.tables.rhythm_counts()


def profile_from_store(
    store: FigureWorkStore,
    *,
    min_n: int,
    max_n: int,
) -> FigureProfile:
    groups: list[FigureProfileGroup] = []
    for key, totals in sorted(figure_group_totals(_iter_store_counts(store)).items()):
        groups.append(
            FigureProfileGroup(
                scale_type=ScaleType(key.scale_type),
                hand=Hand(key.hand),
                n=key.figure_length,
                total=totals.total,
                monophonic=totals.monophonic,
                chords_only=totals.chords_only,
                in_scale=totals.in_scale,
            )
        )

    return FigureProfile(
        metadata=FigureProfileMetadata(min_n=min_n, max_n=max_n, sample_count=store.encoded_sample_count()),
        groups=tuple(groups),
    )


def export_counts_csv(
    store: FigureWorkStore,
    path: Path,
    *,
    limit_per_group: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COUNT_CSV_COLUMNS)
        writer.writeheader()
        for row in store.tables.iter_limited_figure_rows(limit_per_group=limit_per_group):
            writer.writerow(
                {
                    SCALE_TYPE_COLUMN: row.scale_type,
                    HAND_COLUMN: row.hand,
                    N_COLUMN: row.figure_length,
                    COUNT_COLUMN: row.occurrence_count,
                    FIGURE_COLUMN: figure_signature_to_ngram(figure_signature_from_json(row.figure)).model_dump_json(),
                }
            )
    temp_path.replace(path)


def export_base_durations_csv(
    store: FigureWorkStore,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BASE_DURATION_CSV_COLUMNS)
        writer.writeheader()
        for row in store.tables.iter_base_duration_rows():
            writer.writerow(
                {
                    SCALE_TYPE_COLUMN: row.scale_type,
                    HAND_COLUMN: row.hand,
                    N_COLUMN: row.figure_length,
                    BASE_DURATION_COLUMN: row.base_duration,
                    COUNT_COLUMN: row.occurrence_count,
                }
            )
    temp_path.replace(path)


def export_sample_counts(
    store: FigureWorkStore,
    path: Path,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    count = 0
    with temp_path.open("w", encoding="utf-8") as file:
        for payload in store.tables.iter_sample_payloads():
            file.write(payload)
            file.write("\n")
            count += 1
    temp_path.replace(path)
    return count


def write_profile_atomically(profile: FigureProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(profile.model_dump_json(indent=JSON_INDENT), encoding="utf-8")
    temp_path.replace(path)


def copy_file_atomically(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == target_path.resolve():
        return

    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    shutil.copyfile(source_path, temp_path)
    temp_path.replace(target_path)


def _iter_store_counts(store: FigureWorkStore) -> Iterator[tuple[FigureCountKey, int]]:
    yield from store.tables.iter_figure_counts()
