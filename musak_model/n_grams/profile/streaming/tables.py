import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import NamedTuple, cast

from musak_model.n_grams.profile.rhythm.schema import RhythmCountCounter, RhythmCountKey, RhythmMetricKind
from musak_model.n_grams.profile.streaming.schema import FigureCountCounter, FigureCountKey


class FigureCountRow(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int
    figure: str
    occurrence_count: int


class FigureWorkTables:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def configure_connection(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")

    def initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS counts(
                    scale_type TEXT NOT NULL,
                    hand TEXT NOT NULL,
                    n INTEGER NOT NULL,
                    figure TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY(scale_type, hand, n, figure)
                )
                """)
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS sample_counts(
                    sample_index INTEGER PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """)
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS rhythm_counts(
                    scale_type TEXT NOT NULL,
                    time_signature TEXT NOT NULL,
                    hand TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    parameter TEXT NOT NULL,
                    value TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY(scale_type, time_signature, hand, kind, parameter, value)
                )
                """)
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS completed_batches(
                    batch_index INTEGER PRIMARY KEY,
                    sample_start_index INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL
                )
                """)

    def completed_batch_indexes(self) -> set[int]:
        cursor = self._connection.execute("SELECT batch_index FROM completed_batches")
        return {int(row[0]) for row in cursor.fetchall()}

    def add_figure_counts(self, counts: FigureCountCounter) -> None:
        self._connection.executemany(
            """
            INSERT INTO counts(scale_type, hand, n, figure, count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scale_type, hand, n, figure)
            DO UPDATE SET count = count + excluded.count
            """,
            ((key.scale_type, key.hand, key.figure_length, key.figure, count) for key, count in counts.items()),
        )

    def upsert_sample_payloads(self, sample_payloads: Iterable[tuple[int, str]]) -> None:
        self._connection.executemany(
            """
            INSERT INTO sample_counts(sample_index, payload)
            VALUES (?, ?)
            ON CONFLICT(sample_index)
            DO UPDATE SET payload = excluded.payload
            """,
            sample_payloads,
        )

    def add_rhythm_counts(self, counts: RhythmCountCounter) -> None:
        self._connection.executemany(
            """
            INSERT INTO rhythm_counts(scale_type, time_signature, hand, kind, parameter, value, count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scale_type, time_signature, hand, kind, parameter, value)
            DO UPDATE SET count = count + excluded.count
            """,
            (
                (key.scale_type, key.time_signature, key.hand, key.kind, key.parameter, key.value, count)
                for key, count in counts.items()
            ),
        )

    def upsert_completed_batch(
        self,
        *,
        batch_index: int,
        sample_start_index: int,
        sample_count: int,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO completed_batches(batch_index, sample_start_index, sample_count)
            VALUES (?, ?, ?)
            ON CONFLICT(batch_index)
            DO UPDATE SET sample_start_index = excluded.sample_start_index, sample_count = excluded.sample_count
            """,
            (batch_index, sample_start_index, sample_count),
        )

    def metadata_value(self, key: str) -> str | None:
        cursor = self._connection.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return str(row[0]) if row is not None else None

    def set_metadata(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def iter_sample_payloads(self) -> Iterator[str]:
        cursor = self._connection.execute("SELECT payload FROM sample_counts ORDER BY sample_index")
        for (payload,) in cursor:
            yield str(payload)

    def iter_figure_counts(self) -> Iterator[tuple[FigureCountKey, int]]:
        cursor = self._connection.execute("SELECT scale_type, hand, n, figure, count FROM counts")
        for scale_type, hand, figure_length, figure, count in cursor:
            yield (
                FigureCountKey(
                    scale_type=str(scale_type),
                    hand=str(hand),
                    figure_length=int(figure_length),
                    figure=str(figure),
                ),
                int(count),
            )

    def iter_limited_figure_rows(self, *, limit_per_group: int | None) -> Iterator[FigureCountRow]:
        current_group: tuple[str, str, int] | None = None
        current_group_count = 0
        cursor = self._connection.execute("""
            SELECT scale_type, hand, n, figure, count
            FROM counts
            ORDER BY scale_type, hand, n, count DESC, figure
            """)
        for scale_type, hand, figure_length, figure, count in cursor:
            group = (str(scale_type), str(hand), int(figure_length))
            if group != current_group:
                current_group = group
                current_group_count = 0

            if limit_per_group is not None and current_group_count >= limit_per_group:
                continue

            current_group_count += 1
            yield FigureCountRow(
                scale_type=str(scale_type),
                hand=str(hand),
                figure_length=int(figure_length),
                figure=str(figure),
                occurrence_count=int(count),
            )

    def rhythm_counts(self) -> RhythmCountCounter:
        counts: RhythmCountCounter = Counter()
        cursor = self._connection.execute("""
            SELECT scale_type, time_signature, hand, kind, parameter, value, count
            FROM rhythm_counts
            """)
        for scale_type, time_signature, hand, kind, parameter, value, count in cursor:
            counts[
                RhythmCountKey(
                    scale_type=str(scale_type),
                    time_signature=str(time_signature),
                    hand=str(hand),
                    kind=cast(RhythmMetricKind, kind),
                    parameter=str(parameter),
                    value=str(value),
                )
            ] += int(count)

        return counts
