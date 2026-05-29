from pathlib import Path
from typing import Final

import polars as pl

_PARQUET_SUFFIX: Final[str] = ".parquet"
_CSV_SUFFIX: Final[str] = ".csv"


def write_table(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.tmp"
    suffix = path.suffix.lower()
    if suffix == _PARQUET_SUFFIX:
        frame.write_parquet(temp_path)
    elif suffix == _CSV_SUFFIX:
        frame.write_csv(temp_path)
    else:
        raise ValueError(f"unsupported table suffix: {path.suffix}")

    temp_path.replace(path)


def read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == _PARQUET_SUFFIX:
        return pl.read_parquet(path)

    if suffix == _CSV_SUFFIX:
        return pl.read_csv(path)

    raise ValueError(f"unsupported table suffix: {path.suffix}")
