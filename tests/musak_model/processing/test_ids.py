from pathlib import Path

from musak_model.processing.ids import segment_id, source_id


def test_source_and_segment_ids_are_stable_for_relative_source_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "mxl" / "0" / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")

    first_source_id = source_id(source_path, dataset_root=dataset_root)
    second_source_id = source_id(source_path, dataset_root=dataset_root)

    assert first_source_id == second_source_id
    assert segment_id(first_source_id, window_start_bar=8, bar_count=4) == segment_id(
        first_source_id,
        window_start_bar=8,
        bar_count=4,
    )
