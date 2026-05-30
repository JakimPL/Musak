from pathlib import Path

import polars as pl

from musak_model.n_grams.profile.register.schema import (
    REGISTER_ELEMENT_COUNT_COLUMN,
    REGISTER_HAND_COLUMN,
    REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN,
    REGISTER_RESIDUAL_SQUARE_SUM_COLUMN,
    REGISTER_SCALE_TYPE_COLUMN,
    REGISTER_STATISTICS_SCHEMA,
    REGISTER_TREND_SQUARE_SUM_COLUMN,
    RegisterProfileMetadata,
    RegisterStatistics,
    RegisterStatisticsKey,
    RegisterSums,
)
from musak_model.processing.io import JSON_INDENT
from musak_shared.tables import read_table, write_table


def write_register_statistics(statistics: RegisterStatistics, path: Path) -> None:
    records = [
        {
            REGISTER_SCALE_TYPE_COLUMN: key.scale_type,
            REGISTER_HAND_COLUMN: key.hand,
            REGISTER_TREND_SQUARE_SUM_COLUMN: sums.trend_square_sum,
            REGISTER_RESIDUAL_SQUARE_SUM_COLUMN: sums.residual_square_sum,
            REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN: sums.residual_lag_product_sum,
            REGISTER_ELEMENT_COUNT_COLUMN: sums.element_count,
        }
        for key, sums in sorted(statistics.items())
    ]
    write_table(pl.DataFrame(records, schema=REGISTER_STATISTICS_SCHEMA, orient="row"), path)


def read_register_statistics(path: Path) -> RegisterStatistics:
    statistics: RegisterStatistics = {}
    for row in read_table(path).iter_rows(named=True):
        key = RegisterStatisticsKey(
            scale_type=row[REGISTER_SCALE_TYPE_COLUMN],
            hand=row[REGISTER_HAND_COLUMN],
        )
        statistics[key] = RegisterSums(
            trend_square_sum=float(row[REGISTER_TREND_SQUARE_SUM_COLUMN]),
            residual_square_sum=float(row[REGISTER_RESIDUAL_SQUARE_SUM_COLUMN]),
            residual_lag_product_sum=float(row[REGISTER_RESIDUAL_LAG_PRODUCT_SUM_COLUMN]),
            element_count=int(row[REGISTER_ELEMENT_COUNT_COLUMN]),
        )

    return statistics


def write_register_metadata(metadata: RegisterProfileMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.model_dump_json(indent=JSON_INDENT), encoding="utf-8")


def read_register_metadata(path: Path) -> RegisterProfileMetadata:
    return RegisterProfileMetadata.model_validate_json(path.read_text(encoding="utf-8"))
