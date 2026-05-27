from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Self

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from musak_model.n_grams.profile.streaming.tables import FigureWorkTables
from musak_model.tokens.schema import Hand, ScaleType

type AnchorKey = tuple[int, int, int]


class FigureReferenceStore:
    """Read-only access to the durable figure reference database produced by figure extraction."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._engine: Engine | None = None
        self._connection: Connection | None = None
        self._tables: FigureWorkTables | None = None

    def __enter__(self) -> Self:
        if not self.path.is_file():
            raise FileNotFoundError(f"figure reference database does not exist: {self.path}")

        self._engine = create_engine(f"sqlite:///{self.path.resolve().as_posix()}")
        self._connection = self._engine.connect()
        self._tables = FigureWorkTables(self._connection)
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
    def tables(self) -> FigureWorkTables:
        if self._tables is None:
            raise RuntimeError("figure reference store is not open")

        return self._tables

    def figure_counts(
        self,
        *,
        scale_type: ScaleType,
        hand: Hand,
        figure_length: int,
        anchor_degree: int | None = None,
        bar_relative_onset: str | None = None,
    ) -> Counter[str]:
        return self.tables.conditional_figure_counts(
            scale_type=scale_type.value,
            hand=hand.value,
            figure_length=figure_length,
            anchor_degree=anchor_degree,
            bar_relative_onset=bar_relative_onset,
        )

    def base_duration_counts(
        self,
        *,
        scale_type: ScaleType,
        hand: Hand,
        figure_length: int,
    ) -> Counter[str]:
        return self.tables.base_duration_counts(
            scale_type=scale_type.value,
            hand=hand.value,
            figure_length=figure_length,
        )

    def anchor_counts(self, *, scale_type: ScaleType, hand: Hand) -> Counter[AnchorKey]:
        return self.tables.anchor_counts(scale_type=scale_type.value, hand=hand.value)
