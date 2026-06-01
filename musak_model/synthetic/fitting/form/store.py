from collections import Counter
from pathlib import Path
from typing import Final, Self

from sqlalchemy import Boolean, Column, Integer, MetaData, String, Table, select
from sqlalchemy.dialects.sqlite import Insert, insert
from sqlalchemy.engine import Connection, Engine, create_engine

from musak_model.synthetic.fitting.form.statistics import (
    ClosingCounts,
    ClosingKey,
    FormStatistics,
    HistogramCounts,
    HistogramKey,
    PhraseLengthCounts,
    PhraseLengthKey,
    SegmentLengthCounts,
    SegmentLengthKey,
)

_SCALE_TYPE_COLUMN: Final[str] = "scale_type"
_PHRASE_LENGTH_BARS_COLUMN: Final[str] = "phrase_length_bars"
_SEGMENT_LENGTH_BARS_COLUMN: Final[str] = "segment_length_bars"
_IS_FINAL_COLUMN: Final[str] = "is_final"
_FUNCTIONS_COLUMN: Final[str] = "functions"
_BUCKET_COLUMN: Final[str] = "bucket"
_COUNT_COLUMN: Final[str] = "count"
_METADATA_KEY_COLUMN: Final[str] = "key"
_METADATA_VALUE_COLUMN: Final[str] = "value"
_BATCH_INDEX_COLUMN: Final[str] = "batch_index"
_BATCH_SAMPLE_START_INDEX_COLUMN: Final[str] = "sample_start_index"
_BATCH_SAMPLE_COUNT_COLUMN: Final[str] = "sample_count"

_METADATA_STATE_KEY: Final[str] = "state_key"

_SQL_METADATA: Final = MetaData()

_METADATA_TABLE: Final = Table(
    "metadata",
    _SQL_METADATA,
    Column(_METADATA_KEY_COLUMN, String, primary_key=True),
    Column(_METADATA_VALUE_COLUMN, String, nullable=False),
)
_COMPLETED_BATCHES_TABLE: Final = Table(
    "completed_batches",
    _SQL_METADATA,
    Column(_BATCH_INDEX_COLUMN, Integer, primary_key=True),
    Column(_BATCH_SAMPLE_START_INDEX_COLUMN, Integer, nullable=False),
    Column(_BATCH_SAMPLE_COUNT_COLUMN, Integer, nullable=False),
)
_PHRASE_LENGTH_COUNTS_TABLE: Final = Table(
    "phrase_length_counts",
    _SQL_METADATA,
    Column(_SCALE_TYPE_COLUMN, String, primary_key=True),
    Column(_PHRASE_LENGTH_BARS_COLUMN, Integer, primary_key=True),
    Column(_COUNT_COLUMN, Integer, nullable=False),
)
_SEGMENT_LENGTH_COUNTS_TABLE: Final = Table(
    "segment_length_counts",
    _SQL_METADATA,
    Column(_SCALE_TYPE_COLUMN, String, primary_key=True),
    Column(_SEGMENT_LENGTH_BARS_COLUMN, Integer, primary_key=True),
    Column(_COUNT_COLUMN, Integer, nullable=False),
)
_CLOSING_COUNTS_TABLE: Final = Table(
    "closing_counts",
    _SQL_METADATA,
    Column(_SCALE_TYPE_COLUMN, String, primary_key=True),
    Column(_IS_FINAL_COLUMN, Boolean, primary_key=True),
    Column(_FUNCTIONS_COLUMN, String, primary_key=True),
    Column(_COUNT_COLUMN, Integer, nullable=False),
)
_SIMILARITY_HISTOGRAM_TABLE: Final = Table(
    "similarity_histogram",
    _SQL_METADATA,
    Column(_SCALE_TYPE_COLUMN, String, primary_key=True),
    Column(_BUCKET_COLUMN, Integer, primary_key=True),
    Column(_COUNT_COLUMN, Integer, nullable=False),
)
_BEST_MATCH_HISTOGRAM_TABLE: Final = Table(
    "best_match_histogram",
    _SQL_METADATA,
    Column(_SCALE_TYPE_COLUMN, String, primary_key=True),
    Column(_BUCKET_COLUMN, Integer, primary_key=True),
    Column(_COUNT_COLUMN, Integer, nullable=False),
)


class FormWorkTables:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def configure_connection(self) -> None:
        self._connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        self._connection.exec_driver_sql("PRAGMA synchronous=NORMAL")

    def initialize_schema(self) -> None:
        _SQL_METADATA.create_all(self._connection)

    def completed_batch_indexes(self) -> set[int]:
        result = self._connection.execute(select(_COMPLETED_BATCHES_TABLE.c[_BATCH_INDEX_COLUMN]))
        return {int(row[0]) for row in result.fetchall()}

    def add_phrase_length_counts(self, counts: PhraseLengthCounts) -> None:
        records = [
            {
                _SCALE_TYPE_COLUMN: key.scale_type,
                _PHRASE_LENGTH_BARS_COLUMN: key.phrase_length_bars,
                _COUNT_COLUMN: count,
            }
            for key, count in counts.items()
        ]
        if records:
            self._connection.execute(_additive_count_upsert(_PHRASE_LENGTH_COUNTS_TABLE), records)

    def add_segment_length_counts(self, counts: SegmentLengthCounts) -> None:
        records = [
            {
                _SCALE_TYPE_COLUMN: key.scale_type,
                _SEGMENT_LENGTH_BARS_COLUMN: key.segment_length_bars,
                _COUNT_COLUMN: count,
            }
            for key, count in counts.items()
        ]
        if records:
            self._connection.execute(_additive_count_upsert(_SEGMENT_LENGTH_COUNTS_TABLE), records)

    def add_closing_counts(self, counts: ClosingCounts) -> None:
        records = [
            {
                _SCALE_TYPE_COLUMN: key.scale_type,
                _IS_FINAL_COLUMN: key.is_final,
                _FUNCTIONS_COLUMN: key.functions,
                _COUNT_COLUMN: count,
            }
            for key, count in counts.items()
        ]
        if records:
            self._connection.execute(_additive_count_upsert(_CLOSING_COUNTS_TABLE), records)

    def add_similarity_histogram(self, counts: HistogramCounts) -> None:
        self._add_histogram(_SIMILARITY_HISTOGRAM_TABLE, counts)

    def add_best_match_histogram(self, counts: HistogramCounts) -> None:
        self._add_histogram(_BEST_MATCH_HISTOGRAM_TABLE, counts)

    def _add_histogram(self, table: Table, counts: HistogramCounts) -> None:
        records = [
            {_SCALE_TYPE_COLUMN: key.scale_type, _BUCKET_COLUMN: key.bucket, _COUNT_COLUMN: count}
            for key, count in counts.items()
        ]
        if records:
            self._connection.execute(_additive_count_upsert(table), records)

    def upsert_completed_batch(self, *, batch_index: int, sample_start_index: int, sample_count: int) -> None:
        self._connection.execute(
            _replace_upsert(
                _COMPLETED_BATCHES_TABLE,
                update_columns=(_BATCH_SAMPLE_START_INDEX_COLUMN, _BATCH_SAMPLE_COUNT_COLUMN),
            ),
            {
                _BATCH_INDEX_COLUMN: batch_index,
                _BATCH_SAMPLE_START_INDEX_COLUMN: sample_start_index,
                _BATCH_SAMPLE_COUNT_COLUMN: sample_count,
            },
        )

    def metadata_value(self, key: str) -> str | None:
        result = self._connection.execute(
            select(_METADATA_TABLE.c[_METADATA_VALUE_COLUMN]).where(_METADATA_TABLE.c[_METADATA_KEY_COLUMN] == key)
        )
        row = result.fetchone()
        return str(row[0]) if row is not None else None

    def set_metadata(self, key: str, value: str) -> None:
        self._connection.execute(
            _replace_upsert(_METADATA_TABLE, update_columns=(_METADATA_VALUE_COLUMN,)),
            {_METADATA_KEY_COLUMN: key, _METADATA_VALUE_COLUMN: value},
        )

    def phrase_length_counts(self) -> PhraseLengthCounts:
        counts: PhraseLengthCounts = Counter()
        for scale_type, phrase_length_bars, count in self._connection.execute(select(_PHRASE_LENGTH_COUNTS_TABLE)):
            counts[PhraseLengthKey(str(scale_type), int(phrase_length_bars))] += int(count)

        return counts

    def segment_length_counts(self) -> SegmentLengthCounts:
        counts: SegmentLengthCounts = Counter()
        for scale_type, segment_length_bars, count in self._connection.execute(select(_SEGMENT_LENGTH_COUNTS_TABLE)):
            counts[SegmentLengthKey(str(scale_type), int(segment_length_bars))] += int(count)

        return counts

    def closing_counts(self) -> ClosingCounts:
        counts: ClosingCounts = Counter()
        for scale_type, is_final, functions, count in self._connection.execute(select(_CLOSING_COUNTS_TABLE)):
            counts[ClosingKey(str(scale_type), bool(is_final), str(functions))] += int(count)

        return counts

    def similarity_histogram(self) -> HistogramCounts:
        return self._histogram(_SIMILARITY_HISTOGRAM_TABLE)

    def best_match_histogram(self) -> HistogramCounts:
        return self._histogram(_BEST_MATCH_HISTOGRAM_TABLE)

    def _histogram(self, table: Table) -> HistogramCounts:
        counts: HistogramCounts = Counter()
        for scale_type, bucket, count in self._connection.execute(select(table)):
            counts[HistogramKey(str(scale_type), int(bucket))] += int(count)

        return counts


class FormWorkStore:
    def __init__(self, path: Path, *, state_key: str, resume: bool) -> None:
        self.path = path
        self._state_key = state_key
        self._resume = resume
        self._engine: Engine | None = None
        self._connection: Connection | None = None
        self._tables: FormWorkTables | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.path.resolve().as_posix()}")
        self._connection = self._engine.connect()
        self._tables = FormWorkTables(self._connection)
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
            raise RuntimeError("form work store is not open")

        return self._connection

    @property
    def tables(self) -> FormWorkTables:
        if self._tables is None:
            raise RuntimeError("form work store is not open")

        return self._tables

    def completed_batch_indexes(self) -> set[int]:
        return self.tables.completed_batch_indexes()

    def commit_batch(
        self,
        statistics: FormStatistics,
        *,
        batch_index: int,
        sample_start_index: int,
        sample_count: int,
    ) -> None:
        self.connection.commit()
        with self.connection.begin():
            self.tables.add_phrase_length_counts(statistics.phrase_length_counts)
            self.tables.add_segment_length_counts(statistics.segment_length_counts)
            self.tables.add_closing_counts(statistics.closing_counts)
            self.tables.add_similarity_histogram(statistics.similarity_histogram)
            self.tables.add_best_match_histogram(statistics.best_match_histogram)
            self.tables.upsert_completed_batch(
                batch_index=batch_index, sample_start_index=sample_start_index, sample_count=sample_count
            )

    def _validate_or_initialize_state(self) -> None:
        existing_state_key = self.tables.metadata_value(_METADATA_STATE_KEY)
        if existing_state_key is None:
            self.connection.commit()
            with self.connection.begin():
                self.tables.set_metadata(_METADATA_STATE_KEY, self._state_key)
            return

        if existing_state_key != self._state_key:
            raise RuntimeError("partial form extraction state does not match the current analysis configuration")

        _ = self._resume


def _additive_count_upsert(table: Table) -> Insert:
    statement = insert(table)
    return statement.on_conflict_do_update(
        index_elements=list(table.primary_key.columns),
        set_={_COUNT_COLUMN: table.c[_COUNT_COLUMN] + statement.excluded[_COUNT_COLUMN]},
    )


def _replace_upsert(table: Table, *, update_columns: tuple[str, ...]) -> Insert:
    statement = insert(table)
    return statement.on_conflict_do_update(
        index_elements=list(table.primary_key.columns),
        set_={column: statement.excluded[column] for column in update_columns},
    )
