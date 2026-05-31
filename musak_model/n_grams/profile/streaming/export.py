from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.figure.signature import figure_signature_from_json, figure_signature_to_ngram
from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.chord.io import write_chord_metadata, write_chord_transitions, write_figure_by_chord
from musak_model.n_grams.profile.chord.schema import (
    ChordDecodeSpec,
    ChordProfileMetadata,
    FigureByChordCountKey,
    FigureByChordCounts,
    chord_artifact_paths_for_figure_root,
)
from musak_model.n_grams.profile.io import (
    ANCHOR_ACCIDENTAL_COLUMN,
    ANCHOR_DEGREE_COLUMN,
    ANCHOR_FIGURE_COUNT_SCHEMA,
    BASE_DURATION_COLUMN,
    BASE_DURATION_SCHEMA,
    COUNT_COLUMN,
    FIGURE_COLUMN,
    FIGURE_COUNT_SCHEMA,
    HAND_COLUMN,
    N_COLUMN,
    SCALE_TYPE_COLUMN,
)
from musak_model.n_grams.profile.register.io import write_register_metadata, write_register_statistics
from musak_model.n_grams.profile.register.schema import (
    RegisterProfileMetadata,
    register_artifact_paths_for_figure_root,
)
from musak_model.n_grams.profile.rhythm.io import build_rhythm_profile, write_rhythm_counts, write_rhythm_profile
from musak_model.n_grams.profile.rhythm.schema import (
    RhythmCountCounter,
    RhythmProfileMetadata,
    rhythm_artifact_paths_for_figure_root,
)
from musak_model.n_grams.profile.schema import FigureProfile, FigureProfileGroup, FigureProfileMetadata
from musak_model.n_grams.profile.streaming.schema import FigureCountKey, FigureStoreSummary
from musak_model.n_grams.profile.streaming.store import FigureWorkStore
from musak_model.n_grams.profile.streaming.totals import figure_group_totals
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.files import JSON_INDENT, write_yaml_config
from musak_shared.tables import write_table


def export_figure_artifacts(
    store: FigureWorkStore,
    *,
    artifact_paths: FigureArtifactPaths,
    output_path: Path | None,
    config: NGramAnalysisConfig,
    limit_per_group: int | None,
    chord_decode: ChordDecodeSpec | None = None,
) -> FigureStoreSummary:
    profile = profile_from_store(store, min_n=config.figure.min_n, max_n=config.figure.max_n)
    rhythm_counts = rhythm_counts_from_store(store)
    rhythm_paths = rhythm_artifact_paths_for_figure_root(artifact_paths.root_directory)
    rhythm_profile = build_rhythm_profile(
        rhythm_counts,
        metadata=RhythmProfileMetadata(
            rhythm_min_n=config.rhythm.min_n,
            rhythm_max_n=config.rhythm.max_n,
            grid_alignment_denominators=config.rhythm.grid_alignment_denominators,
            strong_beat_offsets=config.rhythm.strong_beat_offsets,
            sample_count=store.encoded_sample_count(),
        ),
    )
    sample_profile_count = export_sample_counts(store, artifact_paths.by_sample_path)
    export_anchor_counts(store, artifact_paths.counts_path)
    export_base_durations(store, artifact_paths.base_durations_path)
    write_rhythm_counts(rhythm_counts, rhythm_paths.counts_path)
    write_rhythm_profile(rhythm_profile, rhythm_paths.profile_path)
    register_paths = register_artifact_paths_for_figure_root(artifact_paths.root_directory)
    write_register_statistics(store.tables.register_statistics(), register_paths.statistics_path)
    write_register_metadata(
        RegisterProfileMetadata(
            arch_basis_count=config.register.arch_basis_count,
            sample_count=store.encoded_sample_count(),
        ),
        register_paths.metadata_path,
    )
    if chord_decode is not None:
        export_chord_artifacts(
            store,
            artifact_paths,
            decode_spec=chord_decode,
            sample_count=store.encoded_sample_count(),
        )
    write_profile_atomically(profile, artifact_paths.profile_path)
    write_yaml_config(config.model_dump(mode="json"), artifact_paths.config_path)
    if output_path is not None:
        export_counts(store, output_path, limit_per_group=limit_per_group)

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


def export_counts(
    store: FigureWorkStore,
    path: Path,
    *,
    limit_per_group: int | None,
) -> None:
    records = [
        {
            SCALE_TYPE_COLUMN: row.scale_type,
            HAND_COLUMN: row.hand,
            N_COLUMN: row.figure_length,
            COUNT_COLUMN: row.occurrence_count,
            FIGURE_COLUMN: figure_signature_to_ngram(figure_signature_from_json(row.figure)).model_dump_json(),
        }
        for row in store.tables.iter_limited_figure_rows(limit_per_group=limit_per_group)
    ]
    write_table(pl.DataFrame(records, schema=FIGURE_COUNT_SCHEMA, orient="row"), path)


def export_anchor_counts(store: FigureWorkStore, path: Path) -> None:
    records = [
        {
            SCALE_TYPE_COLUMN: row.scale_type,
            HAND_COLUMN: row.hand,
            N_COLUMN: row.figure_length,
            ANCHOR_DEGREE_COLUMN: row.anchor_degree,
            ANCHOR_ACCIDENTAL_COLUMN: row.anchor_accidental,
            COUNT_COLUMN: row.occurrence_count,
            FIGURE_COLUMN: figure_signature_to_ngram(figure_signature_from_json(row.figure)).model_dump_json(),
        }
        for row in store.tables.iter_anchor_figure_rows()
    ]
    write_table(pl.DataFrame(records, schema=ANCHOR_FIGURE_COUNT_SCHEMA, orient="row"), path)


def export_chord_artifacts(
    store: FigureWorkStore,
    artifact_paths: FigureArtifactPaths,
    *,
    decode_spec: ChordDecodeSpec,
    sample_count: int,
) -> None:
    chord_paths = chord_artifact_paths_for_figure_root(artifact_paths.root_directory)
    write_chord_transitions(store.tables.chord_transition_counts(), chord_paths.transitions_path)
    write_figure_by_chord(_figure_by_chord_as_ngram(store.tables.figure_by_chord_counts()), chord_paths.figure_path)
    write_chord_metadata(
        ChordProfileMetadata(
            resolution=decode_spec.decoder_config.resolution,
            self_transition_bias=decode_spec.decoder_config.self_transition_bias,
            non_chord_penalty=decode_spec.decoder_config.non_chord_penalty,
            sample_count=sample_count,
        ),
        chord_paths.metadata_path,
    )


def _figure_by_chord_as_ngram(counts: FigureByChordCounts) -> FigureByChordCounts:
    converted: FigureByChordCounts = Counter()
    for key, count in counts.items():
        figure = figure_signature_to_ngram(figure_signature_from_json(key.figure)).model_dump_json()
        converted[FigureByChordCountKey(key.scale_type, key.hand, key.figure_length, key.chord, figure)] += count

    return converted


def export_base_durations(
    store: FigureWorkStore,
    path: Path,
) -> None:
    records = [
        {
            SCALE_TYPE_COLUMN: row.scale_type,
            HAND_COLUMN: row.hand,
            N_COLUMN: row.figure_length,
            BASE_DURATION_COLUMN: row.base_duration,
            COUNT_COLUMN: row.occurrence_count,
        }
        for row in store.tables.iter_base_duration_rows()
    ]
    write_table(pl.DataFrame(records, schema=BASE_DURATION_SCHEMA, orient="row"), path)


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


def _iter_store_counts(store: FigureWorkStore) -> Iterator[tuple[FigureCountKey, int]]:
    yield from store.tables.iter_figure_counts()
