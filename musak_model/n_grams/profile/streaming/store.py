import sqlite3
from pathlib import Path
from typing import Final, Self

from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.io import read_figure_profile
from musak_model.n_grams.profile.streaming.schema import FigureBatchResult, FigureStoreSummary

_WORK_DATABASE_NAME: Final[str] = "work.sqlite3"
_METADATA_STATE_KEY: Final[str] = "state_key"
_METADATA_ENCODED_SAMPLE_COUNT: Final[str] = "encoded_sample_count"


class FigureWorkStore:
    def __init__(
        self,
        path: Path,
        *,
        state_key: str,
        resume: bool,
    ) -> None:
        self.path = path
        self._state_key = state_key
        self._resume = resume
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._initialize_schema()
        self._validate_or_initialize_state()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        if self._connection is not None:
            self._connection.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("figure work store is not open")

        return self._connection

    def completed_batch_indexes(self) -> set[int]:
        cursor = self.connection.execute("SELECT batch_index FROM completed_batches")
        return {int(row[0]) for row in cursor.fetchall()}

    def commit_batch(self, result: FigureBatchResult) -> None:
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO counts(scale_type, hand, n, figure, count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scale_type, hand, n, figure)
                DO UPDATE SET count = count + excluded.count
                """,
                (
                    (scale_type, hand, n, figure, count)
                    for (scale_type, hand, n, figure), count in result.counts.items()
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO sample_counts(sample_index, payload)
                VALUES (?, ?)
                ON CONFLICT(sample_index)
                DO UPDATE SET payload = excluded.payload
                """,
                result.sample_payloads,
            )
            self.connection.execute(
                """
                INSERT INTO completed_batches(batch_index, sample_start_index, sample_count)
                VALUES (?, ?, ?)
                ON CONFLICT(batch_index)
                DO UPDATE SET sample_start_index = excluded.sample_start_index, sample_count = excluded.sample_count
                """,
                (result.batch_index, result.sample_start_index, result.encoded_sample_count),
            )
            self._set_metadata(
                _METADATA_ENCODED_SAMPLE_COUNT,
                str(max(self.encoded_sample_count(), result.sample_start_index + result.encoded_sample_count)),
            )

    def encoded_sample_count(self) -> int:
        value = self._metadata_value(_METADATA_ENCODED_SAMPLE_COUNT)
        return int(value) if value is not None else 0

    def _initialize_schema(self) -> None:
        with self.connection:
            self.connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS counts(
                    scale_type TEXT NOT NULL,
                    hand TEXT NOT NULL,
                    n INTEGER NOT NULL,
                    figure TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY(scale_type, hand, n, figure)
                )
                """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS sample_counts(
                    sample_index INTEGER PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS completed_batches(
                    batch_index INTEGER PRIMARY KEY,
                    sample_start_index INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL
                )
                """)

    def _validate_or_initialize_state(self) -> None:
        existing_state_key = self._metadata_value(_METADATA_STATE_KEY)
        if existing_state_key is None:
            with self.connection:
                self._set_metadata(_METADATA_STATE_KEY, self._state_key)
                self._set_metadata(_METADATA_ENCODED_SAMPLE_COUNT, "0")
            return

        if existing_state_key != self._state_key:
            raise RuntimeError("partial figure extraction state does not match the current analysis configuration")

        _ = self._resume

    def _metadata_value(self, key: str) -> str | None:
        cursor = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return str(row[0]) if row is not None else None

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def figure_work_store_path(artifact_paths: FigureArtifactPaths) -> Path:
    return artifact_paths.root_directory / _WORK_DATABASE_NAME


def clear_figure_work(artifact_paths: FigureArtifactPaths) -> None:
    for path in (
        artifact_paths.config_path,
        artifact_paths.counts_path,
        artifact_paths.profile_path,
        artifact_paths.by_sample_path,
        figure_work_store_path(artifact_paths),
    ):
        path.unlink(missing_ok=True)


def complete_figure_artifacts_exist(artifact_paths: FigureArtifactPaths) -> bool:
    return all(
        path.exists()
        for path in (
            artifact_paths.config_path,
            artifact_paths.counts_path,
            artifact_paths.profile_path,
            artifact_paths.by_sample_path,
        )
    )


def existing_figure_summary(artifact_paths: FigureArtifactPaths) -> FigureStoreSummary:
    profile = read_figure_profile(artifact_paths.profile_path)
    sample_profile_count = sum(
        1 for line in artifact_paths.by_sample_path.read_text(encoding="utf-8").splitlines() if line
    )
    return FigureStoreSummary(
        encoded_sample_count=profile.metadata.sample_count,
        profile_group_count=len(profile.groups),
        sample_profile_count=sample_profile_count,
    )
