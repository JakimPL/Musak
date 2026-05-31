from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd
from sqlalchemy import CheckConstraint, Column, Integer, MetaData, String, Table, and_, create_engine, func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Executable

from musak_model.paths import ARTIFACTS_DIRECTORY
from musak_model.processing.manifest import EncodedManifestField

DEFAULT_DATASET_QUALITY_DATABASE_PATH: Final[Path] = ARTIFACTS_DIRECTORY / "quality" / "dataset_quality.sqlite3"
SQLITE_TRUE: Final[int] = 1
SQLITE_FALSE: Final[int] = 0
_DATASET_NAME_COLUMN: Final[str] = "dataset_name"
_DATASET_ROOT_COLUMN: Final[str] = "dataset_root"
_SOURCE_ID_COLUMN: Final[str] = "source_id"
_SOURCE_PATH_COLUMN: Final[str] = "source_path"
_SOURCE_SHA256_COLUMN: Final[str] = "source_sha256"
_SKIP_FILE_COLUMN: Final[str] = "skip_file"
_SKIPPED_AT_COLUMN: Final[str] = "skipped_at"
_CREATED_AT_COLUMN: Final[str] = "created_at"
_UPDATED_AT_COLUMN: Final[str] = "updated_at"
_WINDOW_START_BAR_COLUMN: Final[str] = "window_start_bar"
_BAR_COUNT_COLUMN: Final[str] = "bar_count"
_RATING_COLUMN: Final[str] = "rating"
_DECISION_COLUMN: Final[str] = "decision"
_TIME_SIGNATURE_ERROR_COLUMN: Final[str] = "time_signature_error"
_KEY_SIGNATURE_ERROR_COLUMN: Final[str] = "key_signature_error"
_MANIFEST_SEGMENT_ID_COLUMN: Final[str] = "manifest_segment_id"
_RATED_AT_COLUMN: Final[str] = "rated_at"

_SQL_METADATA: Final = MetaData()
_REVIEWED_FILES_TABLE: Final = Table(
    "reviewed_files",
    _SQL_METADATA,
    Column(_DATASET_NAME_COLUMN, String, primary_key=True),
    Column(_DATASET_ROOT_COLUMN, String, nullable=False),
    Column(_SOURCE_ID_COLUMN, String, primary_key=True),
    Column(_SOURCE_PATH_COLUMN, String, nullable=False),
    Column(_SOURCE_SHA256_COLUMN, String),
    Column(_SKIP_FILE_COLUMN, Integer, nullable=False, default=SQLITE_FALSE),
    Column(_SKIPPED_AT_COLUMN, String),
    Column(_CREATED_AT_COLUMN, String, nullable=False),
    Column(_UPDATED_AT_COLUMN, String, nullable=False),
    CheckConstraint(f"{_SKIP_FILE_COLUMN} IN (0, 1)"),
)
_SEGMENT_RATINGS_TABLE: Final = Table(
    "segment_ratings",
    _SQL_METADATA,
    Column(_DATASET_NAME_COLUMN, String, primary_key=True),
    Column(_SOURCE_ID_COLUMN, String, primary_key=True),
    Column(_SOURCE_PATH_COLUMN, String, nullable=False),
    Column(_WINDOW_START_BAR_COLUMN, Integer, primary_key=True),
    Column(_BAR_COUNT_COLUMN, Integer, primary_key=True),
    Column(_RATING_COLUMN, Integer, nullable=False),
    Column(_DECISION_COLUMN, String, nullable=False),
    Column(_TIME_SIGNATURE_ERROR_COLUMN, Integer, nullable=False, default=SQLITE_FALSE),
    Column(_KEY_SIGNATURE_ERROR_COLUMN, Integer, nullable=False, default=SQLITE_FALSE),
    Column(_MANIFEST_SEGMENT_ID_COLUMN, String),
    Column(_RATED_AT_COLUMN, String, nullable=False),
    Column(_UPDATED_AT_COLUMN, String, nullable=False),
    CheckConstraint(f"{_RATING_COLUMN} BETWEEN 1 AND 4"),
    CheckConstraint(f"{_DECISION_COLUMN} IN ('OK', 'SKIP', 'TO_CORRECT')"),
    CheckConstraint(f"{_TIME_SIGNATURE_ERROR_COLUMN} IN (0, 1)"),
    CheckConstraint(f"{_KEY_SIGNATURE_ERROR_COLUMN} IN (0, 1)"),
)
_INITIALIZED_DATABASE_PATHS: Final[set[Path]] = set()


class SegmentReviewDecision(StrEnum):
    OK = "OK"
    SKIP = "SKIP"
    TO_CORRECT = "TO_CORRECT"


@dataclass(frozen=True)
class SourceFileReview:
    dataset_name: str
    dataset_root: Path
    source_id: str
    source_path: str
    source_sha256: str | None = None


@dataclass(frozen=True)
class SegmentRating:
    dataset_name: str
    source_id: str
    source_path: str
    window_start_bar: int
    bar_count: int
    rating: int
    decision: SegmentReviewDecision
    time_signature_error: bool
    key_signature_error: bool
    manifest_segment_id: str | None = None


def initialize_quality_database(database_path: Path) -> None:
    resolved_database_path = database_path.resolve()
    if resolved_database_path in _INITIALIZED_DATABASE_PATHS and database_path.exists():
        return

    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _engine(database_path)
    try:
        with engine.begin() as connection:
            _SQL_METADATA.create_all(connection)
        _INITIALIZED_DATABASE_PATHS.add(resolved_database_path)
    finally:
        engine.dispose()


def upsert_segment_rating(database_path: Path, rating: SegmentRating) -> None:
    initialize_quality_database(database_path)
    timestamp = _timestamp()
    record = {
        _DATASET_NAME_COLUMN: rating.dataset_name,
        _SOURCE_ID_COLUMN: rating.source_id,
        _SOURCE_PATH_COLUMN: rating.source_path,
        _WINDOW_START_BAR_COLUMN: rating.window_start_bar,
        _BAR_COUNT_COLUMN: rating.bar_count,
        _RATING_COLUMN: rating.rating,
        _DECISION_COLUMN: rating.decision.value,
        _TIME_SIGNATURE_ERROR_COLUMN: _bool_to_sqlite(rating.time_signature_error),
        _KEY_SIGNATURE_ERROR_COLUMN: _bool_to_sqlite(rating.key_signature_error),
        _MANIFEST_SEGMENT_ID_COLUMN: rating.manifest_segment_id,
        _RATED_AT_COLUMN: timestamp,
        _UPDATED_AT_COLUMN: timestamp,
    }
    statement = insert(_SEGMENT_RATINGS_TABLE).values(record)
    upsert = statement.on_conflict_do_update(
        index_elements=(
            _DATASET_NAME_COLUMN,
            _SOURCE_ID_COLUMN,
            _WINDOW_START_BAR_COLUMN,
            _BAR_COUNT_COLUMN,
        ),
        set_={
            _SOURCE_PATH_COLUMN: statement.excluded.source_path,
            _RATING_COLUMN: statement.excluded.rating,
            _DECISION_COLUMN: statement.excluded.decision,
            _TIME_SIGNATURE_ERROR_COLUMN: statement.excluded.time_signature_error,
            _KEY_SIGNATURE_ERROR_COLUMN: statement.excluded.key_signature_error,
            _MANIFEST_SEGMENT_ID_COLUMN: statement.excluded.manifest_segment_id,
            _RATED_AT_COLUMN: statement.excluded.rated_at,
            _UPDATED_AT_COLUMN: statement.excluded.updated_at,
        },
    )
    _execute(database_path, upsert)


def mark_source_file_skipped(database_path: Path, review: SourceFileReview) -> None:
    initialize_quality_database(database_path)
    timestamp = _timestamp()
    record = {
        _DATASET_NAME_COLUMN: review.dataset_name,
        _DATASET_ROOT_COLUMN: review.dataset_root.as_posix(),
        _SOURCE_ID_COLUMN: review.source_id,
        _SOURCE_PATH_COLUMN: review.source_path,
        _SOURCE_SHA256_COLUMN: review.source_sha256,
        _SKIP_FILE_COLUMN: SQLITE_TRUE,
        _SKIPPED_AT_COLUMN: timestamp,
        _CREATED_AT_COLUMN: timestamp,
        _UPDATED_AT_COLUMN: timestamp,
    }
    statement = insert(_REVIEWED_FILES_TABLE).values(record)
    upsert = statement.on_conflict_do_update(
        index_elements=(_DATASET_NAME_COLUMN, _SOURCE_ID_COLUMN),
        set_={
            _DATASET_ROOT_COLUMN: statement.excluded.dataset_root,
            _SOURCE_PATH_COLUMN: statement.excluded.source_path,
            _SOURCE_SHA256_COLUMN: statement.excluded.source_sha256,
            _SKIP_FILE_COLUMN: SQLITE_TRUE,
            _SKIPPED_AT_COLUMN: statement.excluded.skipped_at,
            _UPDATED_AT_COLUMN: statement.excluded.updated_at,
        },
    )
    _execute(database_path, upsert)


def unrated_source_frame(
    encoded: pd.DataFrame,
    *,
    dataset_name: str,
    database_path: Path,
) -> pd.DataFrame:
    initialize_quality_database(database_path)
    eligible = encoded.loc[_eligible_mask(encoded)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["source_id", "source_path", "eligible_segments", "unrated_segments"])

    skipped_source_ids = _skipped_source_ids(database_path, dataset_name=dataset_name)
    rated_keys = _rated_segment_keys(database_path, dataset_name=dataset_name)
    eligible["_is_skipped"] = eligible[EncodedManifestField.SOURCE_ID].astype(str).isin(skipped_source_ids)
    eligible["_is_rated"] = [
        (
            str(row[EncodedManifestField.SOURCE_ID]),
            _integer_value(row[EncodedManifestField.WINDOW_START_BAR]),
            _integer_value(row[EncodedManifestField.BAR_COUNT]),
        )
        in rated_keys
        for _, row in eligible.iterrows()
    ]
    candidates = eligible.loc[~eligible["_is_skipped"]].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["source_id", "source_path", "eligible_segments", "unrated_segments"])

    summary = (
        candidates.groupby([EncodedManifestField.SOURCE_ID, EncodedManifestField.SOURCE_PATH], dropna=False)
        .agg(
            eligible_segments=(str(EncodedManifestField.SEGMENT_ID), "count"),
            rated_segments=("_is_rated", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                str(EncodedManifestField.SOURCE_ID): "source_id",
                str(EncodedManifestField.SOURCE_PATH): "source_path",
            }
        )
    )
    summary["unrated_segments"] = summary["eligible_segments"] - summary["rated_segments"]
    return summary.loc[
        summary["unrated_segments"] > 0, ["source_id", "source_path", "eligible_segments", "unrated_segments"]
    ]


def eligible_source_rows(encoded: pd.DataFrame, *, source_id: str) -> pd.DataFrame:
    eligible = encoded.loc[_eligible_mask(encoded)].copy()
    return eligible.loc[eligible[EncodedManifestField.SOURCE_ID].astype(str) == source_id].copy()


def rating_by_segment_key(
    database_path: Path,
    *,
    dataset_name: str,
    source_id: str,
) -> dict[tuple[int, int], dict[str, object]]:
    initialize_quality_database(database_path)
    statement = select(
        _SEGMENT_RATINGS_TABLE.c[_WINDOW_START_BAR_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_BAR_COUNT_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_RATING_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_DECISION_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_TIME_SIGNATURE_ERROR_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_KEY_SIGNATURE_ERROR_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_MANIFEST_SEGMENT_ID_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_RATED_AT_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_UPDATED_AT_COLUMN],
    ).where(
        and_(
            _SEGMENT_RATINGS_TABLE.c[_DATASET_NAME_COLUMN] == dataset_name,
            _SEGMENT_RATINGS_TABLE.c[_SOURCE_ID_COLUMN] == source_id,
        )
    )
    rows = _fetch_mappings(database_path, statement)

    return {
        (_integer_value(row[_WINDOW_START_BAR_COLUMN]), _integer_value(row[_BAR_COUNT_COLUMN])): dict(row)
        for row in rows
    }


def approved_segment_rating_rows(
    database_path: Path,
    *,
    dataset_name: str,
    minimum_rating: int = 3,
) -> list[dict[str, object]]:
    initialize_quality_database(database_path)
    ratings = _SEGMENT_RATINGS_TABLE.alias("ratings")
    files = _REVIEWED_FILES_TABLE.alias("files")
    statement = (
        select(ratings)
        .outerjoin(
            files,
            and_(
                files.c[_DATASET_NAME_COLUMN] == ratings.c[_DATASET_NAME_COLUMN],
                files.c[_SOURCE_ID_COLUMN] == ratings.c[_SOURCE_ID_COLUMN],
                files.c[_SKIP_FILE_COLUMN] == SQLITE_TRUE,
            ),
        )
        .where(
            and_(
                ratings.c[_DATASET_NAME_COLUMN] == dataset_name,
                ratings.c[_RATING_COLUMN] >= minimum_rating,
                ratings.c[_DECISION_COLUMN].in_(
                    (SegmentReviewDecision.OK.value, SegmentReviewDecision.TO_CORRECT.value)
                ),
                files.c[_SOURCE_ID_COLUMN].is_(None),
            )
        )
        .order_by(
            ratings.c[_SOURCE_PATH_COLUMN],
            ratings.c[_WINDOW_START_BAR_COLUMN],
            ratings.c[_BAR_COUNT_COLUMN],
        )
    )
    return [dict(row) for row in _fetch_mappings(database_path, statement)]


def quality_database_summary_rows(database_path: Path, *, dataset_name: str) -> list[dict[str, str]]:
    initialize_quality_database(database_path)
    skipped_files = _scalar_count(
        database_path,
        select(func.count())
        .select_from(_REVIEWED_FILES_TABLE)
        .where(
            and_(
                _REVIEWED_FILES_TABLE.c[_DATASET_NAME_COLUMN] == dataset_name,
                _REVIEWED_FILES_TABLE.c[_SKIP_FILE_COLUMN] == SQLITE_TRUE,
            )
        ),
    )
    rated_segments = _scalar_count(
        database_path,
        select(func.count())
        .select_from(_SEGMENT_RATINGS_TABLE)
        .where(_SEGMENT_RATINGS_TABLE.c[_DATASET_NAME_COLUMN] == dataset_name),
    )
    approved_segments = len(approved_segment_rating_rows(database_path, dataset_name=dataset_name))

    return [
        {"Metric": "Skipped files", "Value": str(skipped_files)},
        {"Metric": "Rated segments", "Value": str(rated_segments)},
        {"Metric": "Approved segments", "Value": str(approved_segments)},
    ]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _bool_to_sqlite(value: bool) -> int:
    if not isinstance(value, bool):
        raise TypeError(f"expected bool, got {type(value).__name__}")

    return SQLITE_TRUE if value else SQLITE_FALSE


def _eligible_mask(encoded: pd.DataFrame) -> pd.Series:
    values = encoded[EncodedManifestField.ELIGIBLE_FOR_TRAINING]
    if values.dtype == bool:
        return values

    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _skipped_source_ids(database_path: Path, *, dataset_name: str) -> set[str]:
    statement = select(_REVIEWED_FILES_TABLE.c[_SOURCE_ID_COLUMN]).where(
        and_(
            _REVIEWED_FILES_TABLE.c[_DATASET_NAME_COLUMN] == dataset_name,
            _REVIEWED_FILES_TABLE.c[_SKIP_FILE_COLUMN] == SQLITE_TRUE,
        )
    )
    return {str(row[_SOURCE_ID_COLUMN]) for row in _fetch_mappings(database_path, statement)}


def _rated_segment_keys(database_path: Path, *, dataset_name: str) -> set[tuple[str, int, int]]:
    statement = select(
        _SEGMENT_RATINGS_TABLE.c[_SOURCE_ID_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_WINDOW_START_BAR_COLUMN],
        _SEGMENT_RATINGS_TABLE.c[_BAR_COUNT_COLUMN],
    ).where(_SEGMENT_RATINGS_TABLE.c[_DATASET_NAME_COLUMN] == dataset_name)
    rows = _fetch_mappings(database_path, statement)
    return {
        (
            str(row[_SOURCE_ID_COLUMN]),
            _integer_value(row[_WINDOW_START_BAR_COLUMN]),
            _integer_value(row[_BAR_COUNT_COLUMN]),
        )
        for row in rows
    }


def _integer_value(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def _engine(database_path: Path) -> Engine:
    return create_engine(f"sqlite:///{database_path.resolve().as_posix()}")


def _execute(database_path: Path, statement: Executable) -> None:
    engine = _engine(database_path)
    try:
        with engine.begin() as connection:
            connection.execute(statement)
    finally:
        engine.dispose()


def _fetch_mappings(database_path: Path, statement: Executable) -> list[dict[str, object]]:
    engine = _engine(database_path)
    try:
        with engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]
    finally:
        engine.dispose()


def _scalar_count(database_path: Path, statement: Executable) -> int:
    engine = _engine(database_path)
    try:
        with engine.connect() as connection:
            return int(connection.execute(statement).scalar_one())
    finally:
        engine.dispose()
