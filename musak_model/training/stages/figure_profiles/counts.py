from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from musak_model.n_grams.profile.io import COUNT_COLUMN, FIGURE_COLUMN, HAND_COLUMN, N_COLUMN, SCALE_TYPE_COLUMN
from musak_model.training.stages.figure_profiles.schema import FigureCountGroup
from musak_shared.tables import read_table


def iter_count_groups(path: Path) -> Iterator[FigureCountGroup]:
    frame = read_table(path).sort([SCALE_TYPE_COLUMN, HAND_COLUMN, N_COLUMN])
    current_key: tuple[str, str, int] | None = None
    counts: Counter[str] = Counter()
    for row in frame.iter_rows(named=True):
        key = (row[SCALE_TYPE_COLUMN], row[HAND_COLUMN], int(row[N_COLUMN]))
        if current_key is not None and key != current_key:
            yield FigureCountGroup(key=current_key, counts=counts)
            counts = Counter()

        current_key = key
        counts[row[FIGURE_COLUMN]] += int(row[COUNT_COLUMN])

    if current_key is not None:
        yield FigureCountGroup(key=current_key, counts=counts)
