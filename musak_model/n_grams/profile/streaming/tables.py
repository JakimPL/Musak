from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Final, NamedTuple, cast

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, func, select
from sqlalchemy.dialects.sqlite import Insert, insert
from sqlalchemy.engine import Connection

from musak_model.n_grams.profile.register.schema import RegisterStatistics, RegisterStatisticsKey, RegisterSums
from musak_model.n_grams.profile.rhythm.schema import RhythmCountCounter, RhythmCountKey, RhythmMetricKind
from musak_model.n_grams.profile.streaming.schema import FigureCountCounter, FigureCountKey

_METADATA_KEY_COLUMN: Final[str] = "key"
_METADATA_VALUE_COLUMN: Final[str] = "value"
_COUNT_SCALE_TYPE_COLUMN: Final[str] = "scale_type"
_COUNT_HAND_COLUMN: Final[str] = "hand"
_COUNT_N_COLUMN: Final[str] = "n"
_COUNT_FIGURE_COLUMN: Final[str] = "figure"
_COUNT_ANCHOR_DEGREE_COLUMN: Final[str] = "anchor_degree"
_COUNT_ANCHOR_ACCIDENTAL_COLUMN: Final[str] = "anchor_accidental"
_COUNT_ANCHOR_OCTAVE_COLUMN: Final[str] = "anchor_octave"
_COUNT_BASE_DURATION_COLUMN: Final[str] = "base_duration"
_COUNT_BAR_RELATIVE_ONSET_COLUMN: Final[str] = "bar_relative_onset"
_COUNT_TIME_SIGNATURE_COLUMN: Final[str] = "time_signature"
_COUNT_COUNT_COLUMN: Final[str] = "count"
_SAMPLE_INDEX_COLUMN: Final[str] = "sample_index"
_SAMPLE_PAYLOAD_COLUMN: Final[str] = "payload"
_RHYTHM_TIME_SIGNATURE_COLUMN: Final[str] = "time_signature"
_RHYTHM_KIND_COLUMN: Final[str] = "kind"
_RHYTHM_PARAMETER_COLUMN: Final[str] = "parameter"
_RHYTHM_VALUE_COLUMN: Final[str] = "value"
_REGISTER_TREND_SQUARE_SUM_COLUMN: Final[str] = "trend_square_sum"
_REGISTER_RESIDUAL_SQUARE_SUM_COLUMN: Final[str] = "residual_square_sum"
_REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN: Final[str] = "residual_lag_product_sum"
_REGISTER_ELEMENT_COUNT_COLUMN: Final[str] = "element_count"
_REGISTER_SUM_COLUMNS: Final[tuple[str, ...]] = (
    _REGISTER_TREND_SQUARE_SUM_COLUMN,
    _REGISTER_RESIDUAL_SQUARE_SUM_COLUMN,
    _REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN,
    _REGISTER_ELEMENT_COUNT_COLUMN,
)
_BATCH_INDEX_COLUMN: Final[str] = "batch_index"
_BATCH_SAMPLE_START_INDEX_COLUMN: Final[str] = "sample_start_index"
_BATCH_SAMPLE_COUNT_COLUMN: Final[str] = "sample_count"

_SQL_METADATA: Final = MetaData()

_METADATA_TABLE: Final = Table(
    "metadata",
    _SQL_METADATA,
    Column(_METADATA_KEY_COLUMN, String, primary_key=True),
    Column(_METADATA_VALUE_COLUMN, String, nullable=False),
)
_COUNTS_TABLE: Final = Table(
    "counts",
    _SQL_METADATA,
    Column(_COUNT_SCALE_TYPE_COLUMN, String, primary_key=True),
    Column(_COUNT_HAND_COLUMN, String, primary_key=True),
    Column(_COUNT_N_COLUMN, Integer, primary_key=True),
    Column(_COUNT_FIGURE_COLUMN, String, primary_key=True),
    Column(_COUNT_ANCHOR_DEGREE_COLUMN, Integer, primary_key=True),
    Column(_COUNT_ANCHOR_ACCIDENTAL_COLUMN, Integer, primary_key=True),
    Column(_COUNT_ANCHOR_OCTAVE_COLUMN, Integer, primary_key=True),
    Column(_COUNT_BASE_DURATION_COLUMN, String, primary_key=True),
    Column(_COUNT_BAR_RELATIVE_ONSET_COLUMN, String, primary_key=True),
    Column(_COUNT_TIME_SIGNATURE_COLUMN, String, primary_key=True),
    Column(_COUNT_COUNT_COLUMN, Integer, nullable=False),
)
_SAMPLE_COUNTS_TABLE: Final = Table(
    "sample_counts",
    _SQL_METADATA,
    Column(_SAMPLE_INDEX_COLUMN, Integer, primary_key=True),
    Column(_SAMPLE_PAYLOAD_COLUMN, String, nullable=False),
)
_RHYTHM_COUNTS_TABLE: Final = Table(
    "rhythm_counts",
    _SQL_METADATA,
    Column(_COUNT_SCALE_TYPE_COLUMN, String, primary_key=True),
    Column(_RHYTHM_TIME_SIGNATURE_COLUMN, String, primary_key=True),
    Column(_COUNT_HAND_COLUMN, String, primary_key=True),
    Column(_RHYTHM_KIND_COLUMN, String, primary_key=True),
    Column(_RHYTHM_PARAMETER_COLUMN, String, primary_key=True),
    Column(_RHYTHM_VALUE_COLUMN, String, primary_key=True),
    Column(_COUNT_COUNT_COLUMN, Integer, nullable=False),
)
_REGISTER_STATISTICS_TABLE: Final = Table(
    "register_statistics",
    _SQL_METADATA,
    Column(_COUNT_SCALE_TYPE_COLUMN, String, primary_key=True),
    Column(_COUNT_HAND_COLUMN, String, primary_key=True),
    Column(_REGISTER_TREND_SQUARE_SUM_COLUMN, Float, nullable=False),
    Column(_REGISTER_RESIDUAL_SQUARE_SUM_COLUMN, Float, nullable=False),
    Column(_REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN, Float, nullable=False),
    Column(_REGISTER_ELEMENT_COUNT_COLUMN, Integer, nullable=False),
)
_COMPLETED_BATCHES_TABLE: Final = Table(
    "completed_batches",
    _SQL_METADATA,
    Column(_BATCH_INDEX_COLUMN, Integer, primary_key=True),
    Column(_BATCH_SAMPLE_START_INDEX_COLUMN, Integer, nullable=False),
    Column(_BATCH_SAMPLE_COUNT_COLUMN, Integer, nullable=False),
)


class FigureCountRow(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int
    figure: str
    occurrence_count: int


class AnchorFigureCountRow(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int
    anchor_degree: int
    anchor_accidental: int
    figure: str
    occurrence_count: int


class BaseDurationRow(NamedTuple):
    scale_type: str
    hand: str
    figure_length: int
    base_duration: str
    occurrence_count: int


class FigureWorkTables:
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

    def add_figure_counts(self, counts: FigureCountCounter) -> None:
        records = [
            {
                _COUNT_SCALE_TYPE_COLUMN: key.scale_type,
                _COUNT_HAND_COLUMN: key.hand,
                _COUNT_N_COLUMN: key.figure_length,
                _COUNT_FIGURE_COLUMN: key.figure,
                _COUNT_ANCHOR_DEGREE_COLUMN: key.anchor_degree,
                _COUNT_ANCHOR_ACCIDENTAL_COLUMN: key.anchor_accidental,
                _COUNT_ANCHOR_OCTAVE_COLUMN: key.anchor_octave,
                _COUNT_BASE_DURATION_COLUMN: key.base_duration,
                _COUNT_BAR_RELATIVE_ONSET_COLUMN: key.bar_relative_onset,
                _COUNT_TIME_SIGNATURE_COLUMN: key.time_signature,
                _COUNT_COUNT_COLUMN: count,
            }
            for key, count in counts.items()
        ]
        if records:
            self._connection.execute(_additive_count_upsert(_COUNTS_TABLE), records)

    def upsert_sample_payloads(self, sample_payloads: Iterable[tuple[int, str]]) -> None:
        records = [
            {
                _SAMPLE_INDEX_COLUMN: sample_index,
                _SAMPLE_PAYLOAD_COLUMN: payload,
            }
            for sample_index, payload in sample_payloads
        ]
        if records:
            self._connection.execute(
                _replace_upsert(_SAMPLE_COUNTS_TABLE, update_columns=(_SAMPLE_PAYLOAD_COLUMN,)),
                records,
            )

    def add_rhythm_counts(self, counts: RhythmCountCounter) -> None:
        records = [
            {
                _COUNT_SCALE_TYPE_COLUMN: key.scale_type,
                _RHYTHM_TIME_SIGNATURE_COLUMN: key.time_signature,
                _COUNT_HAND_COLUMN: key.hand,
                _RHYTHM_KIND_COLUMN: key.kind,
                _RHYTHM_PARAMETER_COLUMN: key.parameter,
                _RHYTHM_VALUE_COLUMN: key.value,
                _COUNT_COUNT_COLUMN: count,
            }
            for key, count in counts.items()
        ]
        if records:
            self._connection.execute(_additive_count_upsert(_RHYTHM_COUNTS_TABLE), records)

    def add_register_statistics(self, statistics: RegisterStatistics) -> None:
        records = [
            {
                _COUNT_SCALE_TYPE_COLUMN: key.scale_type,
                _COUNT_HAND_COLUMN: key.hand,
                _REGISTER_TREND_SQUARE_SUM_COLUMN: sums.trend_square_sum,
                _REGISTER_RESIDUAL_SQUARE_SUM_COLUMN: sums.residual_square_sum,
                _REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN: sums.residual_lag_product_sum,
                _REGISTER_ELEMENT_COUNT_COLUMN: sums.element_count,
            }
            for key, sums in statistics.items()
        ]
        if records:
            self._connection.execute(_additive_register_upsert(_REGISTER_STATISTICS_TABLE), records)

    def upsert_completed_batch(
        self,
        *,
        batch_index: int,
        sample_start_index: int,
        sample_count: int,
    ) -> None:
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
            _replace_upsert(
                _METADATA_TABLE,
                update_columns=(_METADATA_VALUE_COLUMN,),
            ),
            {
                _METADATA_KEY_COLUMN: key,
                _METADATA_VALUE_COLUMN: value,
            },
        )

    def iter_sample_payloads(self) -> Iterator[str]:
        result = self._connection.execute(
            select(_SAMPLE_COUNTS_TABLE.c[_SAMPLE_PAYLOAD_COLUMN]).order_by(
                _SAMPLE_COUNTS_TABLE.c[_SAMPLE_INDEX_COLUMN]
            )
        )
        for (payload,) in result:
            yield str(payload)

    def iter_figure_counts(self) -> Iterator[tuple[FigureCountKey, int]]:
        result = self._connection.execute(select(_COUNTS_TABLE))
        for row in result.mappings():
            yield (
                FigureCountKey(
                    scale_type=str(row[_COUNT_SCALE_TYPE_COLUMN]),
                    hand=str(row[_COUNT_HAND_COLUMN]),
                    figure_length=int(row[_COUNT_N_COLUMN]),
                    figure=str(row[_COUNT_FIGURE_COLUMN]),
                    anchor_degree=int(row[_COUNT_ANCHOR_DEGREE_COLUMN]),
                    anchor_accidental=int(row[_COUNT_ANCHOR_ACCIDENTAL_COLUMN]),
                    anchor_octave=int(row[_COUNT_ANCHOR_OCTAVE_COLUMN]),
                    base_duration=str(row[_COUNT_BASE_DURATION_COLUMN]),
                    bar_relative_onset=str(row[_COUNT_BAR_RELATIVE_ONSET_COLUMN]),
                    time_signature=str(row[_COUNT_TIME_SIGNATURE_COLUMN]),
                ),
                int(row[_COUNT_COUNT_COLUMN]),
            )

    def iter_limited_figure_rows(self, *, limit_per_group: int | None) -> Iterator[FigureCountRow]:
        current_group: tuple[str, str, int] | None = None
        current_group_count = 0
        aggregated_count = func.sum(_COUNTS_TABLE.c[_COUNT_COUNT_COLUMN]).label(_COUNT_COUNT_COLUMN)
        result = self._connection.execute(
            select(
                _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN],
                _COUNTS_TABLE.c[_COUNT_HAND_COLUMN],
                _COUNTS_TABLE.c[_COUNT_N_COLUMN],
                _COUNTS_TABLE.c[_COUNT_FIGURE_COLUMN],
                aggregated_count,
            )
            .group_by(
                _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN],
                _COUNTS_TABLE.c[_COUNT_HAND_COLUMN],
                _COUNTS_TABLE.c[_COUNT_N_COLUMN],
                _COUNTS_TABLE.c[_COUNT_FIGURE_COLUMN],
            )
            .order_by(
                _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN],
                _COUNTS_TABLE.c[_COUNT_HAND_COLUMN],
                _COUNTS_TABLE.c[_COUNT_N_COLUMN],
                aggregated_count.desc(),
                _COUNTS_TABLE.c[_COUNT_FIGURE_COLUMN],
            )
        )
        for scale_type, hand, figure_length, figure, count in result:
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

    def iter_anchor_figure_rows(self) -> Iterator[AnchorFigureCountRow]:
        aggregated_count = func.sum(_COUNTS_TABLE.c[_COUNT_COUNT_COLUMN]).label(_COUNT_COUNT_COLUMN)
        group_columns = (
            _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN],
            _COUNTS_TABLE.c[_COUNT_HAND_COLUMN],
            _COUNTS_TABLE.c[_COUNT_N_COLUMN],
            _COUNTS_TABLE.c[_COUNT_ANCHOR_DEGREE_COLUMN],
            _COUNTS_TABLE.c[_COUNT_ANCHOR_ACCIDENTAL_COLUMN],
            _COUNTS_TABLE.c[_COUNT_FIGURE_COLUMN],
        )
        result = self._connection.execute(
            select(*group_columns, aggregated_count).group_by(*group_columns).order_by(*group_columns)
        )
        for scale_type, hand, figure_length, anchor_degree, anchor_accidental, figure, count in result:
            yield AnchorFigureCountRow(
                scale_type=str(scale_type),
                hand=str(hand),
                figure_length=int(figure_length),
                anchor_degree=int(anchor_degree),
                anchor_accidental=int(anchor_accidental),
                figure=str(figure),
                occurrence_count=int(count),
            )

    def rhythm_counts(self) -> RhythmCountCounter:
        counts: RhythmCountCounter = Counter()
        result = self._connection.execute(select(_RHYTHM_COUNTS_TABLE))
        for scale_type, time_signature, hand, kind, parameter, value, count in result:
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

    def register_statistics(self) -> RegisterStatistics:
        statistics: RegisterStatistics = {}
        result = self._connection.execute(select(_REGISTER_STATISTICS_TABLE))
        for scale_type, hand, trend_square_sum, residual_square_sum, residual_lag_product_sum, element_count in result:
            statistics[RegisterStatisticsKey(scale_type=str(scale_type), hand=str(hand))] = RegisterSums(
                trend_square_sum=float(trend_square_sum),
                residual_square_sum=float(residual_square_sum),
                residual_lag_product_sum=float(residual_lag_product_sum),
                element_count=int(element_count),
            )

        return statistics

    def conditional_figure_counts(
        self,
        *,
        scale_type: str,
        hand: str,
        figure_length: int,
        anchor_degree: int | None = None,
        bar_relative_onset: str | None = None,
    ) -> Counter[str]:
        aggregated_count = func.sum(_COUNTS_TABLE.c[_COUNT_COUNT_COLUMN]).label(_COUNT_COUNT_COLUMN)
        conditions = [
            _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN] == scale_type,
            _COUNTS_TABLE.c[_COUNT_HAND_COLUMN] == hand,
            _COUNTS_TABLE.c[_COUNT_N_COLUMN] == figure_length,
        ]
        if anchor_degree is not None:
            conditions.append(_COUNTS_TABLE.c[_COUNT_ANCHOR_DEGREE_COLUMN] == anchor_degree)
        if bar_relative_onset is not None:
            conditions.append(_COUNTS_TABLE.c[_COUNT_BAR_RELATIVE_ONSET_COLUMN] == bar_relative_onset)

        statement = (
            select(_COUNTS_TABLE.c[_COUNT_FIGURE_COLUMN], aggregated_count)
            .where(*conditions)
            .group_by(_COUNTS_TABLE.c[_COUNT_FIGURE_COLUMN])
        )
        counts: Counter[str] = Counter()
        for figure, count in self._connection.execute(statement):
            counts[str(figure)] += int(count)

        return counts

    def base_duration_counts(self, *, scale_type: str, hand: str, figure_length: int) -> Counter[str]:
        aggregated_count = func.sum(_COUNTS_TABLE.c[_COUNT_COUNT_COLUMN]).label(_COUNT_COUNT_COLUMN)
        statement = (
            select(_COUNTS_TABLE.c[_COUNT_BASE_DURATION_COLUMN], aggregated_count)
            .where(
                _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN] == scale_type,
                _COUNTS_TABLE.c[_COUNT_HAND_COLUMN] == hand,
                _COUNTS_TABLE.c[_COUNT_N_COLUMN] == figure_length,
            )
            .group_by(_COUNTS_TABLE.c[_COUNT_BASE_DURATION_COLUMN])
        )
        counts: Counter[str] = Counter()
        for base_duration, count in self._connection.execute(statement):
            counts[str(base_duration)] += int(count)

        return counts

    def iter_base_duration_rows(self) -> Iterator[BaseDurationRow]:
        aggregated_count = func.sum(_COUNTS_TABLE.c[_COUNT_COUNT_COLUMN]).label(_COUNT_COUNT_COLUMN)
        result = self._connection.execute(
            select(
                _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN],
                _COUNTS_TABLE.c[_COUNT_HAND_COLUMN],
                _COUNTS_TABLE.c[_COUNT_N_COLUMN],
                _COUNTS_TABLE.c[_COUNT_BASE_DURATION_COLUMN],
                aggregated_count,
            )
            .group_by(
                _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN],
                _COUNTS_TABLE.c[_COUNT_HAND_COLUMN],
                _COUNTS_TABLE.c[_COUNT_N_COLUMN],
                _COUNTS_TABLE.c[_COUNT_BASE_DURATION_COLUMN],
            )
            .order_by(
                _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN],
                _COUNTS_TABLE.c[_COUNT_HAND_COLUMN],
                _COUNTS_TABLE.c[_COUNT_N_COLUMN],
                _COUNTS_TABLE.c[_COUNT_BASE_DURATION_COLUMN],
            )
        )
        for scale_type, hand, figure_length, base_duration, count in result:
            yield BaseDurationRow(
                scale_type=str(scale_type),
                hand=str(hand),
                figure_length=int(figure_length),
                base_duration=str(base_duration),
                occurrence_count=int(count),
            )

    def anchor_counts(self, *, scale_type: str, hand: str) -> Counter[tuple[int, int, int]]:
        aggregated_count = func.sum(_COUNTS_TABLE.c[_COUNT_COUNT_COLUMN]).label(_COUNT_COUNT_COLUMN)
        statement = (
            select(
                _COUNTS_TABLE.c[_COUNT_ANCHOR_DEGREE_COLUMN],
                _COUNTS_TABLE.c[_COUNT_ANCHOR_ACCIDENTAL_COLUMN],
                _COUNTS_TABLE.c[_COUNT_ANCHOR_OCTAVE_COLUMN],
                aggregated_count,
            )
            .where(
                _COUNTS_TABLE.c[_COUNT_SCALE_TYPE_COLUMN] == scale_type,
                _COUNTS_TABLE.c[_COUNT_HAND_COLUMN] == hand,
            )
            .group_by(
                _COUNTS_TABLE.c[_COUNT_ANCHOR_DEGREE_COLUMN],
                _COUNTS_TABLE.c[_COUNT_ANCHOR_ACCIDENTAL_COLUMN],
                _COUNTS_TABLE.c[_COUNT_ANCHOR_OCTAVE_COLUMN],
            )
        )
        counts: Counter[tuple[int, int, int]] = Counter()
        for anchor_degree, anchor_accidental, anchor_octave, count in self._connection.execute(statement):
            counts[(int(anchor_degree), int(anchor_accidental), int(anchor_octave))] += int(count)

        return counts


def _additive_count_upsert(table: Table) -> Insert:
    statement = insert(table)
    return statement.on_conflict_do_update(
        index_elements=list(table.primary_key.columns),
        set_={
            _COUNT_COUNT_COLUMN: table.c[_COUNT_COUNT_COLUMN] + statement.excluded[_COUNT_COUNT_COLUMN],
        },
    )


def _additive_register_upsert(table: Table) -> Insert:
    statement = insert(table)
    return statement.on_conflict_do_update(
        index_elements=list(table.primary_key.columns),
        set_={column: table.c[column] + statement.excluded[column] for column in _REGISTER_SUM_COLUMNS},
    )


def _replace_upsert(table: Table, *, update_columns: tuple[str, ...]) -> Insert:
    statement = insert(table)
    return statement.on_conflict_do_update(
        index_elements=list(table.primary_key.columns),
        set_={column: statement.excluded[column] for column in update_columns},
    )
