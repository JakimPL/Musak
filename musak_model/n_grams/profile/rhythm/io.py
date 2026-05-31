from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import polars as pl

from musak_model.n_grams.profile.rhythm.schema import (
    RHYTHM_COUNT_COLUMN,
    RHYTHM_COUNT_SCHEMA,
    RHYTHM_HAND_COLUMN,
    RHYTHM_KIND_COLUMN,
    RHYTHM_PARAMETER_COLUMN,
    RHYTHM_SCALE_TYPE_COLUMN,
    RHYTHM_TIME_SIGNATURE_COLUMN,
    RHYTHM_VALUE_COLUMN,
    RhythmCountCounter,
    RhythmCountKey,
    RhythmGroupKey,
    RhythmMetricKind,
    RhythmProfile,
    RhythmProfileGroup,
    RhythmProfileMetadata,
)
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.files import JSON_INDENT
from musak_shared.tables import read_table, write_table


def write_rhythm_counts(counts: RhythmCountCounter, path: Path) -> None:
    records = [
        {
            RHYTHM_SCALE_TYPE_COLUMN: key.scale_type,
            RHYTHM_TIME_SIGNATURE_COLUMN: key.time_signature,
            RHYTHM_HAND_COLUMN: key.hand,
            RHYTHM_KIND_COLUMN: key.kind,
            RHYTHM_PARAMETER_COLUMN: key.parameter,
            RHYTHM_VALUE_COLUMN: key.value,
            RHYTHM_COUNT_COLUMN: count,
        }
        for key, count in sorted(counts.items())
    ]
    write_table(pl.DataFrame(records, schema=RHYTHM_COUNT_SCHEMA, orient="row"), path)


def read_rhythm_counts(path: Path) -> RhythmCountCounter:
    counts: RhythmCountCounter = Counter()
    for row in read_table(path).iter_rows(named=True):
        key = RhythmCountKey(
            scale_type=row[RHYTHM_SCALE_TYPE_COLUMN],
            time_signature=row[RHYTHM_TIME_SIGNATURE_COLUMN],
            hand=row[RHYTHM_HAND_COLUMN],
            kind=cast(RhythmMetricKind, row[RHYTHM_KIND_COLUMN]),
            parameter=row[RHYTHM_PARAMETER_COLUMN],
            value=row[RHYTHM_VALUE_COLUMN],
        )
        counts[key] += int(row[RHYTHM_COUNT_COLUMN])

    return counts


def build_rhythm_profile(
    counts: RhythmCountCounter,
    *,
    metadata: RhythmProfileMetadata,
) -> RhythmProfile:
    groups: list[RhythmProfileGroup] = []
    for key, group_counts in _iter_group_counts(counts):
        groups.append(
            RhythmProfileGroup(
                scale_type=ScaleType(key.scale_type),
                time_signature=key.time_signature,
                hand=Hand(key.hand),
                kind=key.kind,
                parameter=key.parameter,
                total=sum(group_counts.values()),
                unique_values=len(group_counts),
            )
        )

    return RhythmProfile(metadata=metadata, groups=tuple(groups))


def write_rhythm_profile(profile: RhythmProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=JSON_INDENT), encoding="utf-8")


def read_rhythm_profile(path: Path) -> RhythmProfile:
    return RhythmProfile.model_validate_json(path.read_text(encoding="utf-8"))


def _iter_group_counts(
    counts: RhythmCountCounter,
) -> Iterable[tuple[RhythmGroupKey, Counter[str]]]:
    groups: dict[RhythmGroupKey, Counter[str]] = {}
    for key, count in counts.items():
        group_key = RhythmGroupKey(
            scale_type=key.scale_type,
            time_signature=key.time_signature,
            hand=key.hand,
            kind=key.kind,
            parameter=key.parameter,
        )
        groups.setdefault(group_key, Counter())[key.value] += count

    return sorted(groups.items())
