from pathlib import Path

import pytest

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.tokens.schema import BarToken, DurationClass, EndToken, Hand, NoteToken, ScaleType, Token
from musak_model.training import ingestion
from musak_model.training.ingestion import split as ingestion_split


def _note() -> NoteToken:
    return NoteToken(
        degree=1,
        accidental=0,
        octave_offset=0,
        duration=DurationClass.QUARTER,
    )


def _segment(source_file: Path) -> Segment:
    tokens: list[Token] = [_note(), BarToken(), _note(), BarToken(), EndToken()]
    return Segment(
        right_hand_tokens=tokens,
        left_hand_tokens=tokens,
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            source_file=source_file,
            difficulty_level=2,
        ),
    )


def _ingestion_config(*, split_seed: int, validation_fraction: float) -> ingestion.IngestionConfig:
    return ingestion.IngestionConfig(
        window_bars=8,
        stride_bars=4,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        difficulty_labels=None,
    )


def test_build_ingestion_split_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_paths = [tmp_path / f"piece_{index}.mxl" for index in range(4)]
    for file_path in file_paths:
        file_path.write_text("score")

    def fake_process_file(
        path: Path,
        *,
        window_bars: int,
        stride_bars: int,
        difficulty_labels: dict[str, int] | None,
    ) -> list[Segment]:
        return [_segment(path)]

    monkeypatch.setattr("musak_model.training.ingestion.split.process_file", fake_process_file)

    config = _ingestion_config(split_seed=123, validation_fraction=0.25)
    split_a = ingestion.build_split(tmp_path, config=config)
    split_b = ingestion.build_split(tmp_path, config=config)

    assert [sample.source_file for sample in split_a.validation] == [
        sample.source_file for sample in split_b.validation
    ]
    assert len(split_a.train) + len(split_a.validation) == 8
    assert split_a.invalid_files == []


def test_build_ingestion_split_collects_invalid_file_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good_file = tmp_path / "good.mxl"
    bad_file = tmp_path / "bad.mxl"
    good_file.write_text("score")
    bad_file.write_text("score")

    def fake_process_file(
        path: Path,
        *,
        window_bars: int,
        stride_bars: int,
        difficulty_labels: dict[str, int] | None,
    ) -> list[Segment]:
        if path.name == "bad.mxl":
            raise ValueError("parse failed")
        return [_segment(path)]

    monkeypatch.setattr("musak_model.training.ingestion.split.process_file", fake_process_file)

    config = _ingestion_config(split_seed=7, validation_fraction=0.5)
    split = ingestion.build_split(tmp_path, config=config)

    assert len(split.invalid_files) == 1
    assert split.invalid_files[0].file.endswith("bad.mxl")
    assert split.invalid_files[0].exception_type == "ValueError"
    assert split.invalid_files[0].message == "parse failed"
    assert all(sample.source_file.name == "good.mxl" for sample in split.train + split.validation)


def test_build_bar_positions_from_tokens_assigns_end_to_last_bar() -> None:
    tokens: list[Token] = [_note(), BarToken(), _note(), BarToken(), EndToken()]

    bar_positions = ingestion_split._build_bar_positions_from_tokens(tokens)

    assert bar_positions == [0, 0, 1, 1, 1]


def test_encode_segment_hands_returns_both_hands() -> None:
    segment = _segment(Path("piece.mxl"))

    samples = ingestion_split._encode_segment_hands(segment)

    assert len(samples) == 2
    assert {sample.hand for sample in samples} == {Hand.RIGHT, Hand.LEFT}
    assert all(len(sample.token_ids) == len(sample.bar_positions) for sample in samples)


def test_load_ingestion_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "ingestion.yml"
    config_path.write_text(
        "\n".join(
            [
                "window_bars: 8",
                "stride_bars: 4",
                "validation_fraction: 0.25",
                "split_seed: 99",
                "difficulty_labels:",
                "  sample: 3",
            ]
        )
    )

    config = ingestion.IngestionConfig.load_config(config_path)

    assert config.window_bars == 8
    assert config.stride_bars == 4
    assert config.validation_fraction == 0.25
    assert config.split_seed == 99
    assert config.difficulty_labels == {"sample": 3}


def test_ingestion_config_rejects_invalid_validation_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        ingestion.IngestionConfig(
            window_bars=8,
            stride_bars=4,
            validation_fraction=1.0,
            split_seed=17,
            difficulty_labels=None,
        )
