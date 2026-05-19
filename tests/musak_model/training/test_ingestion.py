from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.data.config import SegmentationConfig
from musak_model.data.schema import Segment, SegmentIneligibilityReason, SegmentMetadata
from musak_model.processing.dataset import process_dataset
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, NoteToken, ScaleType, Token
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.split import _build_bar_positions_from_tokens, _encode_segment, build_split
from tests.musak_model.data.fixtures import bar, note_event, parsed_score


def _note(duration_vocabulary: DurationVocabulary) -> NoteToken:
    quarter_duration_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    return NoteToken(
        degree=1,
        accidental=0,
        octave_offset=0,
        duration_id=quarter_duration_id,
    )


def _segment(source_file: Path, *, duration_vocabulary: DurationVocabulary) -> Segment:
    tokens: list[Token] = [
        _note(duration_vocabulary),
        BarToken(),
        _note(duration_vocabulary),
        BarToken(),
        EndToken(),
    ]
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            window_start_bar=0,
            source_file=source_file,
            difficulty_level=2,
        ),
    )


def _ineligible_segment(source_file: Path, *, duration_vocabulary: DurationVocabulary) -> Segment:
    segment = _segment(source_file, duration_vocabulary=duration_vocabulary)
    metadata = segment.metadata.model_copy(
        update={
            "eligible_for_training": False,
            "ineligibility_reasons": frozenset({SegmentIneligibilityReason.MIXED_TIME_SIGNATURE}),
        }
    )
    return segment.model_copy(update={"metadata": metadata})


def _ingestion_config(*, split_seed: int, validation_fraction: float) -> IngestionConfig:
    return IngestionConfig(
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        difficulty_labels=None,
    )


def _segmentation_config() -> SegmentationConfig:
    return SegmentationConfig(window_bars=8, stride_bars=4)


def test_build_ingestion_split_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duration_vocabulary: DurationVocabulary,
    tokenization_config: TokenizationConfig,
) -> None:
    file_paths = [tmp_path / f"piece_{index}.mxl" for index in range(4)]
    for file_path in file_paths:
        file_path.write_text("score")

    def fake_process_file(
        path: Path,
        *,
        segmentation: SegmentationConfig,
        difficulty_labels: dict[str, int] | None,
        duration_vocabulary: DurationVocabulary | None = None,
    ) -> list[Segment]:
        assert duration_vocabulary is not None
        return [_segment(path, duration_vocabulary=duration_vocabulary)]

    monkeypatch.setattr("musak_model.training.ingestion.split.process_file", fake_process_file)

    config = _ingestion_config(split_seed=123, validation_fraction=0.25)
    split_a = build_split(
        tmp_path,
        config=config,
        segmentation=_segmentation_config(),
        tokenization_config=tokenization_config,
    )
    split_b = build_split(
        tmp_path,
        config=config,
        segmentation=_segmentation_config(),
        tokenization_config=tokenization_config,
    )

    assert [sample.source_file for sample in split_a.validation] == [
        sample.source_file for sample in split_b.validation
    ]
    assert len(split_a.train) + len(split_a.validation) == 4
    assert split_a.invalid_files == []


def test_build_ingestion_split_collects_invalid_file_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    good_file = tmp_path / "good.mxl"
    bad_file = tmp_path / "bad.mxl"
    good_file.write_text("score")
    bad_file.write_text("score")

    def fake_process_file(
        path: Path,
        *,
        segmentation: SegmentationConfig,
        difficulty_labels: dict[str, int] | None,
        duration_vocabulary: DurationVocabulary | None = None,
    ) -> list[Segment]:
        if path.name == "bad.mxl":
            raise ValueError("parse failed")
        assert duration_vocabulary is not None
        return [_segment(path, duration_vocabulary=duration_vocabulary)]

    monkeypatch.setattr("musak_model.training.ingestion.split.process_file", fake_process_file)

    config = _ingestion_config(split_seed=7, validation_fraction=0.5)
    split = build_split(
        tmp_path,
        config=config,
        segmentation=_segmentation_config(),
        tokenization_config=tokenization_config,
    )

    assert len(split.invalid_files) == 1
    assert split.invalid_files[0].file.endswith("bad.mxl")
    assert split.invalid_files[0].exception_type == "ValueError"
    assert split.invalid_files[0].message == "parse failed"
    assert all(sample.source_file.name == "good.mxl" for sample in split.train + split.validation)


def test_build_bar_positions_from_tokens_assigns_end_to_last_bar(duration_vocabulary: DurationVocabulary) -> None:
    tokens: list[Token] = [
        _note(duration_vocabulary),
        BarToken(),
        _note(duration_vocabulary),
        BarToken(),
        EndToken(),
    ]

    bar_positions = _build_bar_positions_from_tokens(tokens)

    assert bar_positions == [0, 0, 1, 1, 1]


def test_encode_segment_returns_unified_sample(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    segment = _segment(Path("piece.mxl"), duration_vocabulary=duration_vocabulary)
    sample = _encode_segment(segment, token_vocabulary=token_vocabulary)

    assert sample.hand is None
    assert len(sample.token_ids) == len(sample.bar_positions)


def test_build_ingestion_split_filters_ineligible_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duration_vocabulary: DurationVocabulary,
    tokenization_config: TokenizationConfig,
) -> None:
    file_path = tmp_path / "piece.mxl"
    file_path.write_text("score")

    def fake_process_file(
        path: Path,
        *,
        segmentation: SegmentationConfig,
        difficulty_labels: dict[str, int] | None,
        duration_vocabulary: DurationVocabulary | None = None,
    ) -> list[Segment]:
        assert duration_vocabulary is not None
        return [
            _segment(path, duration_vocabulary=duration_vocabulary),
            _ineligible_segment(path, duration_vocabulary=duration_vocabulary),
        ]

    monkeypatch.setattr("musak_model.training.ingestion.split.process_file", fake_process_file)

    config = _ingestion_config(split_seed=123, validation_fraction=0.0)
    split = build_split(
        tmp_path,
        config=config,
        segmentation=_segmentation_config(),
        tokenization_config=tokenization_config,
    )

    assert len(split.train) == 1
    assert split.invalid_files == []


def test_build_ingestion_split_prefers_encoded_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    processed_root = tmp_path / "processed"
    score = parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
    )
    monkeypatch.setattr("musak_model.processing.dataset.parse_score", lambda path: score)
    process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
        stage="all",
        overwrite=True,
    )
    monkeypatch.setattr(
        "musak_model.training.ingestion.split.process_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw processing should not run")),
    )

    split = build_split(
        dataset_root,
        config=IngestionConfig(
            validation_fraction=0.0,
            split_seed=17,
            difficulty_labels=None,
            processed_root=processed_root,
        ),
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
    )

    assert len(split.train) == 1
    assert split.invalid_files == []


def test_build_ingestion_split_ignores_encoded_artifacts_without_matching_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    processed_root = tmp_path / "processed"
    score = parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
    )
    monkeypatch.setattr("musak_model.processing.dataset.parse_score", lambda path: score)
    result = process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
        stage="all",
        overwrite=True,
    )
    assert result.tokenizer_snapshot_path is not None
    result.tokenizer_snapshot_path.unlink()
    monkeypatch.setattr(
        "musak_model.training.ingestion.split.process_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw processing should not run")),
    )

    split = build_split(
        dataset_root,
        config=IngestionConfig(
            validation_fraction=0.0,
            split_seed=17,
            difficulty_labels=None,
            processed_root=processed_root,
        ),
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
    )

    assert len(split.train) == 1
    assert split.invalid_files == []


def test_build_ingestion_split_falls_back_to_parsed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    dataset_root = tmp_path / "PDMX"
    source_path = dataset_root / "piece.mxl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("score")
    processed_root = tmp_path / "processed"
    score = parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
    )
    monkeypatch.setattr("musak_model.processing.dataset.parse_score", lambda path: score)
    process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
        stage="parsed",
        overwrite=True,
    )
    monkeypatch.setattr(
        "musak_model.training.ingestion.split.process_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw processing should not run")),
    )

    split = build_split(
        dataset_root,
        config=IngestionConfig(
            validation_fraction=0.0,
            split_seed=17,
            difficulty_labels=None,
            processed_root=processed_root,
        ),
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
    )

    assert len(split.train) == 1
    assert split.invalid_files == []


def test_build_ingestion_split_requires_raw_fallback_when_processed_artifacts_are_unusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenization_config: TokenizationConfig,
) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    monkeypatch.setattr(
        "musak_model.training.ingestion.split.process_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw processing should not run")),
    )

    with pytest.raises(ValueError, match="raw MusicXML fallback is disabled"):
        build_split(
            Path("PDMX"),
            config=IngestionConfig(
                validation_fraction=0.0,
                split_seed=17,
                difficulty_labels=None,
                processed_root=processed_root,
            ),
            segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
            tokenization_config=tokenization_config,
            allow_raw_fallback=False,
        )


def test_load_ingestion_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "ingestion.yml"
    config_path.write_text(
        "\n".join(
            [
                "validation_fraction: 0.25",
                "split_seed: 99",
                "difficulty_labels:",
                "  sample: 3",
            ]
        )
    )

    config = IngestionConfig.load(config_path)

    assert config.validation_fraction == 0.25
    assert config.split_seed == 99
    assert config.difficulty_labels == {"sample": 3}


def test_ingestion_config_rejects_invalid_validation_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        IngestionConfig(
            validation_fraction=1.0,
            split_seed=17,
            difficulty_labels=None,
        )
