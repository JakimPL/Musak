from collections import Counter
from pathlib import Path

import pytest

from musak_model.synthetic.fitting.form.io import form_artifact_paths_for_figure_root, read_form_statistics
from musak_model.synthetic.fitting.form.orchestration import export_form_statistics
from musak_model.synthetic.fitting.form.statistics import ClosingKey, FormStatistics, PhraseLengthKey
from musak_model.synthetic.fitting.form.store import FormWorkStore


def _statistics() -> FormStatistics:
    return FormStatistics(
        phrase_length_counts=Counter({PhraseLengthKey("major", 4): 1}),
        closing_counts=Counter({ClosingKey("major", True, "dominant>tonic"): 1}),
    )


def test_store_accumulates_across_batches_and_resumes(tmp_path: Path) -> None:
    database_path = tmp_path / "form.sqlite3"
    with FormWorkStore(database_path, state_key="state", resume=False) as store:
        store.commit_batch(_statistics(), batch_index=0, sample_start_index=0, sample_count=1)

    with FormWorkStore(database_path, state_key="state", resume=True) as store:
        assert store.completed_batch_indexes() == {0}
        store.commit_batch(_statistics(), batch_index=1, sample_start_index=1, sample_count=1)

        assert store.tables.phrase_length_counts()[PhraseLengthKey("major", 4)] == 2
        assert store.tables.closing_counts()[ClosingKey("major", True, "dominant>tonic")] == 2
        assert store.completed_batch_indexes() == {0, 1}


def test_store_rejects_a_mismatched_state_key(tmp_path: Path) -> None:
    database_path = tmp_path / "form.sqlite3"
    with FormWorkStore(database_path, state_key="first", resume=False):
        pass

    with pytest.raises(RuntimeError, match="does not match"):
        with FormWorkStore(database_path, state_key="second", resume=False):
            pass


def test_export_round_trips_through_parquet(tmp_path: Path) -> None:
    artifact_paths = form_artifact_paths_for_figure_root(tmp_path)
    with FormWorkStore(artifact_paths.database_path, state_key="state", resume=False) as store:
        store.commit_batch(_statistics(), batch_index=0, sample_start_index=0, sample_count=1)
        export_form_statistics(store, artifact_paths)

    restored = read_form_statistics(artifact_paths)

    assert restored is not None
    assert restored.phrase_length_counts[PhraseLengthKey("major", 4)] == 1
    assert restored.closing_counts[ClosingKey("major", True, "dominant>tonic")] == 1
