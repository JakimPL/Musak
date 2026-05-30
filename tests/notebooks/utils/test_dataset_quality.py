from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from musak_model.processing.manifest import EncodedManifestField
from notebooks.utils.dataset_quality import (
    SegmentRating,
    SegmentReviewDecision,
    SourceFileReview,
    approved_segment_rating_rows,
    initialize_quality_database,
    mark_source_file_skipped,
    unrated_source_frame,
    upsert_segment_rating,
)


def test_initialize_quality_database_creates_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "quality.sqlite3"

    initialize_quality_database(database_path)

    with sqlite3.connect(database_path) as connection:
        table_names = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert {"reviewed_files", "segment_ratings"}.issubset(table_names)


def test_upsert_segment_rating_inserts_and_updates(tmp_path: Path) -> None:
    database_path = tmp_path / "quality.sqlite3"

    upsert_segment_rating(database_path, _rating(rating=2, decision=SegmentReviewDecision.SKIP))
    upsert_segment_rating(database_path, _rating(rating=4, decision=SegmentReviewDecision.OK))

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT rating, decision FROM segment_ratings WHERE dataset_name = 'PDMX'").fetchall()

    assert rows == [(4, SegmentReviewDecision.OK.value)]


def test_segment_rating_constraints_reject_invalid_values(tmp_path: Path) -> None:
    database_path = tmp_path / "quality.sqlite3"
    initialize_quality_database(database_path)

    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("""
                INSERT INTO segment_ratings (
                    dataset_name,
                    source_id,
                    source_path,
                    window_start_bar,
                    bar_count,
                    rating,
                    decision,
                    time_signature_error,
                    key_signature_error,
                    rated_at,
                    updated_at
                )
                VALUES ('PDMX', 'source-a', 'a.mxl', 0, 8, 5, 'OK', 0, 0, 'now', 'now')
                """)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("""
                INSERT INTO segment_ratings (
                    dataset_name,
                    source_id,
                    source_path,
                    window_start_bar,
                    bar_count,
                    rating,
                    decision,
                    time_signature_error,
                    key_signature_error,
                    rated_at,
                    updated_at
                )
                VALUES ('PDMX', 'source-a', 'a.mxl', 0, 8, 3, 'MAYBE', 0, 0, 'now', 'now')
                """)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("""
                INSERT INTO segment_ratings (
                    dataset_name,
                    source_id,
                    source_path,
                    window_start_bar,
                    bar_count,
                    rating,
                    decision,
                    time_signature_error,
                    key_signature_error,
                    rated_at,
                    updated_at
                )
                VALUES ('PDMX', 'source-a', 'a.mxl', 0, 8, 3, 'OK', 2, 0, 'now', 'now')
                """)


def test_mark_source_file_skipped(tmp_path: Path) -> None:
    database_path = tmp_path / "quality.sqlite3"

    mark_source_file_skipped(
        database_path,
        SourceFileReview(
            dataset_name="PDMX",
            dataset_root=tmp_path / "data" / "PDMX",
            source_id="source-a",
            source_path="a.mxl",
            source_sha256="abc",
        ),
    )

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT skip_file, source_sha256 FROM reviewed_files").fetchall()

    assert rows == [(1, "abc")]


def test_unrated_source_frame_excludes_skipped_and_fully_rated_files(tmp_path: Path) -> None:
    database_path = tmp_path / "quality.sqlite3"
    frame = _encoded_frame()
    upsert_segment_rating(database_path, _rating(source_id="source-a", source_path="a.mxl", start=0))
    upsert_segment_rating(database_path, _rating(source_id="source-a", source_path="a.mxl", start=8))
    mark_source_file_skipped(
        database_path,
        SourceFileReview(
            dataset_name="PDMX",
            dataset_root=tmp_path / "data" / "PDMX",
            source_id="source-c",
            source_path="c.mxl",
        ),
    )

    unrated = unrated_source_frame(frame, dataset_name="PDMX", database_path=database_path)

    assert unrated["source_id"].tolist() == ["source-b"]
    assert unrated["unrated_segments"].tolist() == [1]


def test_unrated_source_frame_keeps_partially_rated_files(tmp_path: Path) -> None:
    database_path = tmp_path / "quality.sqlite3"
    frame = _encoded_frame()
    upsert_segment_rating(database_path, _rating(source_id="source-a", source_path="a.mxl", start=0))

    unrated = unrated_source_frame(frame, dataset_name="PDMX", database_path=database_path)

    row = unrated.loc[unrated["source_id"] == "source-a"].iloc[0]
    assert row["eligible_segments"] == 2
    assert row["unrated_segments"] == 1


def test_approved_segment_rating_rows_apply_default_filter_and_file_skip(tmp_path: Path) -> None:
    database_path = tmp_path / "quality.sqlite3"
    upsert_segment_rating(database_path, _rating(source_id="low", source_path="low.mxl", rating=2))
    upsert_segment_rating(
        database_path,
        _rating(source_id="skip", source_path="skip.mxl", rating=4, decision=SegmentReviewDecision.SKIP),
    )
    upsert_segment_rating(
        database_path,
        _rating(source_id="correct", source_path="correct.mxl", rating=3, decision=SegmentReviewDecision.TO_CORRECT),
    )
    upsert_segment_rating(database_path, _rating(source_id="ok", source_path="ok.mxl", rating=4))
    upsert_segment_rating(database_path, _rating(source_id="file-skip", source_path="file-skip.mxl", rating=4))
    mark_source_file_skipped(
        database_path,
        SourceFileReview(
            dataset_name="PDMX",
            dataset_root=tmp_path / "data" / "PDMX",
            source_id="file-skip",
            source_path="file-skip.mxl",
        ),
    )

    approved = approved_segment_rating_rows(database_path, dataset_name="PDMX")

    assert [row["source_id"] for row in approved] == ["correct", "ok"]


def _rating(
    *,
    source_id: str = "source-a",
    source_path: str = "a.mxl",
    start: int = 0,
    rating: int = 3,
    decision: SegmentReviewDecision = SegmentReviewDecision.OK,
) -> SegmentRating:
    return SegmentRating(
        dataset_name="PDMX",
        source_id=source_id,
        source_path=source_path,
        window_start_bar=start,
        bar_count=8,
        rating=rating,
        decision=decision,
        time_signature_error=False,
        key_signature_error=False,
        manifest_segment_id=f"{source_id}-{start}",
    )


def _encoded_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _encoded_row(source_id="source-a", source_path="a.mxl", start=0),
            _encoded_row(source_id="source-a", source_path="a.mxl", start=8),
            _encoded_row(source_id="source-b", source_path="b.mxl", start=0),
            _encoded_row(source_id="source-c", source_path="c.mxl", start=0),
            _encoded_row(source_id="source-d", source_path="d.mxl", start=0, eligible=False),
        ]
    )


def _encoded_row(
    *,
    source_id: str,
    source_path: str,
    start: int,
    eligible: bool = True,
) -> dict[str, object]:
    return {
        EncodedManifestField.SEGMENT_ID.value: f"{source_id}-{start}",
        EncodedManifestField.SOURCE_ID.value: source_id,
        EncodedManifestField.SOURCE_PATH.value: source_path,
        EncodedManifestField.WINDOW_START_BAR.value: start,
        EncodedManifestField.BAR_COUNT.value: 8,
        EncodedManifestField.ELIGIBLE_FOR_TRAINING.value: eligible,
    }
