import csv
import shutil
from collections.abc import Iterator
from pathlib import Path

from musak_model.n_grams.figure.signature import figure_signature_from_json, figure_signature_to_ngram
from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.io import COUNT_CSV_COLUMNS
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
    limit_per_group: int | None,
) -> FigureStoreSummary:
    profile = profile_from_store(store, min_n=min_n, max_n=max_n)
    sample_profile_count = export_sample_counts(store, artifact_paths.by_sample_path)
    export_counts_csv(store, artifact_paths.counts_path, limit_per_group=None)
    write_profile_atomically(profile, artifact_paths.profile_path)
    copy_file_atomically(analysis_config_path, artifact_paths.config_path)
    if output_path is not None:
        export_counts_csv(store, output_path, limit_per_group=limit_per_group)

    return FigureStoreSummary(
        encoded_sample_count=store.encoded_sample_count(),
        profile_group_count=len(profile.groups),
        sample_profile_count=sample_profile_count,
    )


def profile_from_store(
    store: FigureWorkStore,
    *,
    min_n: int,
    max_n: int,
) -> FigureProfile:
    groups: list[FigureProfileGroup] = []
    for (scale_type, hand, figure_length), totals in sorted(figure_group_totals(_iter_store_counts(store)).items()):
        groups.append(
            FigureProfileGroup(
                scale_type=ScaleType(scale_type),
                hand=Hand(hand),
                n=figure_length,
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
        for scale_type, hand, figure_length, figure_json, count in _iter_limited_store_rows(
            store,
            limit_per_group=limit_per_group,
        ):
            writer.writerow(
                {
                    "scale_type": scale_type,
                    "hand": hand,
                    "n": figure_length,
                    "count": count,
                    "figure": figure_signature_to_ngram(figure_signature_from_json(figure_json)).model_dump_json(),
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
        cursor = store.connection.execute("SELECT payload FROM sample_counts ORDER BY sample_index")
        for (payload,) in cursor:
            file.write(str(payload))
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
    cursor = store.connection.execute("SELECT scale_type, hand, n, figure, count FROM counts")
    for scale_type, hand, figure_length, figure, count in cursor:
        yield (str(scale_type), str(hand), int(figure_length), str(figure)), int(count)


def _iter_limited_store_rows(
    store: FigureWorkStore,
    *,
    limit_per_group: int | None,
) -> Iterator[tuple[str, str, int, str, int]]:
    query = """
        SELECT scale_type, hand, n, figure, count
        FROM counts
        ORDER BY scale_type, hand, n, count DESC, figure
    """
    current_group: tuple[str, str, int] | None = None
    current_group_count = 0
    cursor = store.connection.execute(query)
    for scale_type, hand, figure_length, figure, count in cursor:
        group = (str(scale_type), str(hand), int(figure_length))
        if group != current_group:
            current_group = group
            current_group_count = 0

        if limit_per_group is not None and current_group_count >= limit_per_group:
            continue

        current_group_count += 1
        yield str(scale_type), str(hand), int(figure_length), str(figure), int(count)
