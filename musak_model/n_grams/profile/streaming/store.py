from pathlib import Path
from typing import Final, Self

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.io import read_figure_profile
from musak_model.n_grams.profile.rhythm.schema import rhythm_artifact_paths_for_figure_root
from musak_model.n_grams.profile.streaming.schema import FigureBatchResult, FigureStoreSummary
from musak_model.n_grams.profile.streaming.tables import FigureWorkTables

_REFERENCE_DATABASE_NAME: Final[str] = "figures.sqlite3"
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
        self._engine: Engine | None = None
        self._connection: Connection | None = None
        self._tables: FigureWorkTables | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.path.resolve().as_posix()}")
        self._connection = self._engine.connect()
        self._tables = FigureWorkTables(self._connection)
        self.tables.configure_connection()
        self.tables.initialize_schema()
        self.connection.commit()
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
        if self._engine is not None:
            self._engine.dispose()

    @property
    def connection(self) -> Connection:
        if self._connection is None:
            raise RuntimeError("figure work store is not open")

        return self._connection

    @property
    def tables(self) -> FigureWorkTables:
        if self._tables is None:
            raise RuntimeError("figure work store is not open")

        return self._tables

    def completed_batch_indexes(self) -> set[int]:
        return self.tables.completed_batch_indexes()

    def commit_batch(self, result: FigureBatchResult) -> None:
        self.connection.commit()
        with self.connection.begin():
            self.tables.add_figure_counts(result.counts)
            self.tables.upsert_sample_payloads(result.sample_payloads)
            self.tables.add_rhythm_counts(result.rhythm_counts)
            self.tables.upsert_completed_batch(
                batch_index=result.batch_index,
                sample_start_index=result.sample_start_index,
                sample_count=result.encoded_sample_count,
            )
            self._set_metadata(
                _METADATA_ENCODED_SAMPLE_COUNT,
                str(max(self.encoded_sample_count(), result.sample_start_index + result.encoded_sample_count)),
            )

    def encoded_sample_count(self) -> int:
        value = self._metadata_value(_METADATA_ENCODED_SAMPLE_COUNT)
        return int(value) if value is not None else 0

    def _validate_or_initialize_state(self) -> None:
        existing_state_key = self._metadata_value(_METADATA_STATE_KEY)
        if existing_state_key is None:
            self.connection.commit()
            with self.connection.begin():
                self._set_metadata(_METADATA_STATE_KEY, self._state_key)
                self._set_metadata(_METADATA_ENCODED_SAMPLE_COUNT, "0")
            return

        if existing_state_key != self._state_key:
            raise RuntimeError("partial figure extraction state does not match the current analysis configuration")

        _ = self._resume

    def _metadata_value(self, key: str) -> str | None:
        return self.tables.metadata_value(key)

    def _set_metadata(self, key: str, value: str) -> None:
        self.tables.set_metadata(key, value)


def figure_reference_database_path(artifact_paths: FigureArtifactPaths) -> Path:
    return artifact_paths.root_directory / _REFERENCE_DATABASE_NAME


def clear_figure_work(artifact_paths: FigureArtifactPaths) -> None:
    rhythm_paths = rhythm_artifact_paths_for_figure_root(artifact_paths.root_directory)
    for path in (
        artifact_paths.config_path,
        artifact_paths.counts_path,
        artifact_paths.profile_path,
        artifact_paths.by_sample_path,
        rhythm_paths.counts_path,
        rhythm_paths.profile_path,
        figure_reference_database_path(artifact_paths),
    ):
        path.unlink(missing_ok=True)


def complete_figure_artifacts_exist(artifact_paths: FigureArtifactPaths) -> bool:
    return all(
        path.exists()
        for path in (
            artifact_paths.config_path,
            artifact_paths.counts_path,
            artifact_paths.base_durations_path,
            artifact_paths.profile_path,
            artifact_paths.by_sample_path,
        )
    )


def complete_reference_artifacts_exist(artifact_paths: FigureArtifactPaths) -> bool:
    rhythm_paths = rhythm_artifact_paths_for_figure_root(artifact_paths.root_directory)
    return complete_figure_artifacts_exist(artifact_paths) and all(
        path.exists()
        for path in (
            rhythm_paths.counts_path,
            rhythm_paths.profile_path,
            figure_reference_database_path(artifact_paths),
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
