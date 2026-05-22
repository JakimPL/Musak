from fractions import Fraction
from pathlib import Path

import pytest
from music21 import instrument
from music21.meter.base import TimeSignature
from music21.note import Note
from music21.stream.base import Measure, Part, Score

from musak_model.conditioning.time_signature import TimeSignatureVocabulary, TimeSignatureVocabularyConfig
from musak_model.data.config import DataProcessingConfig, SegmentationConfig, SegmentationMode
from musak_model.data.schema import Segment, SegmentIneligibilityReason, SegmentMetadata
from musak_model.processing.dataset import process_dataset
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, NoteToken, RestToken, ScaleType, Token
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.dataset import EncodedExerciseDataset, collate_training_examples
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.split import _build_bar_positions_from_tokens, _encode_segment, build_split
from musak_model.training.validity import TrainingValidityMaskBuilder
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
            scale_root=0,
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
        duration_vocabulary: DurationVocabulary,
        *,
        segmentation_config: SegmentationConfig,
        difficulty_labels: dict[str, int] | None,
    ) -> list[Segment]:
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
        duration_vocabulary: DurationVocabulary,
        *,
        segmentation_config: SegmentationConfig,
        difficulty_labels: dict[str, int] | None,
    ) -> list[Segment]:
        if path.name == "bad.mxl":
            raise ValueError("parse failed")
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
        duration_vocabulary: DurationVocabulary,
        *,
        segmentation_config: SegmentationConfig,
        difficulty_labels: dict[str, int] | None,
    ) -> list[Segment]:
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
    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: score)
    process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="process",
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


def test_build_ingestion_split_rejects_windowed_encoded_artifacts_for_whole_file_training(
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
    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: score)
    process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1, mode=SegmentationMode.WINDOWED),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="process",
        overwrite=True,
    )

    with pytest.raises(ValueError, match="whole-file segmentation requires encoded artifacts"):
        build_split(
            dataset_root,
            config=IngestionConfig(
                validation_fraction=0.0,
                split_seed=17,
                difficulty_labels=None,
                processed_root=processed_root,
            ),
            segmentation=SegmentationConfig(window_bars=1, stride_bars=1, mode=SegmentationMode.WHOLE_FILE),
            tokenization_config=tokenization_config,
        )


@pytest.mark.parametrize(
    ("short_bar_position", "expected_bar_durations"),
    [
        ("pickup", (Fraction(1, 4), Fraction(1, 1))),
        ("final", (Fraction(1, 1), Fraction(1, 4))),
    ],
)
def test_raw_ingestion_pipeline_preserves_short_measure_bar_tokens(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    token_vocabulary: TokenVocabulary,
    short_bar_position: str,
    expected_bar_durations: tuple[Fraction, Fraction],
) -> None:
    score_path = tmp_path / f"{short_bar_position}.musicxml"
    _write_short_measure_score(score_path, short_bar_position=short_bar_position)

    split = build_split(
        tmp_path,
        config=IngestionConfig(validation_fraction=0.0, split_seed=17),
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1, mode=SegmentationMode.WHOLE_FILE),
        tokenization_config=tokenization_config,
    )

    assert len(split.train) == 1
    sample = split.train[0]
    tokens = token_vocabulary.decode(sample.token_ids)
    assert sample.metadata.bar_durations == expected_bar_durations
    assert sum(isinstance(token, BarToken) for token in tokens) == 2
    assert not _has_rest_duration(tokens, Fraction(3, 4), token_vocabulary=token_vocabulary)

    dataset = EncodedExerciseDataset(
        [sample],
        time_signature_vocabulary=TimeSignatureVocabulary(
            TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2)
        ),
        token_vocabulary=token_vocabulary,
    )
    batch = collate_training_examples([dataset[0]])
    masks = TrainingValidityMaskBuilder(token_vocabulary).masks_for_batch(batch, device=batch.input_token_ids.device)

    assert not masks.invalid_target_mask.any().item()


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
    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: score)
    result = process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="process",
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
    monkeypatch.setattr("musak_model.processing.parse.parse_score", lambda path: score)
    process_dataset(
        dataset_root,
        processed_root=processed_root,
        segmentation_config=SegmentationConfig(window_bars=1, stride_bars=1),
        tokenization_config=tokenization_config,
        data_processing_config=DataProcessingConfig(remove_segments_with_silent_bars=True),
        stage="parse",
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


def _write_short_measure_score(path: Path, *, short_bar_position: str) -> None:
    score = Score()
    right = Part()
    right.insert(0, instrument.Piano())
    left = Part()
    left.insert(0, instrument.Piano())

    for part, short_pitch, full_pitch in ((right, "C5", "D5"), (left, "C3", "D3")):
        match short_bar_position:
            case "pickup":
                part.append(_measure(0, short_pitch, quarter_length=1, include_time_signature=True))
                part.append(_measure(1, full_pitch, quarter_length=4))
            case "final":
                part.append(_measure(1, full_pitch, quarter_length=4, include_time_signature=True))
                part.append(_measure(2, short_pitch, quarter_length=1))
            case _:
                raise ValueError(f"unsupported short_bar_position: {short_bar_position}")

    score.insert(0, right)
    score.insert(0, left)
    score.write("musicxml", fp=path)


def _measure(
    number: int,
    pitch_name: str,
    *,
    quarter_length: int,
    include_time_signature: bool = False,
) -> Measure:
    measure = Measure(number=number)
    if include_time_signature:
        measure.insert(0, TimeSignature("4/4"))

    measure.insert(0, Note(pitch_name, quarterLength=quarter_length))
    if quarter_length < 4:
        measure.paddingRight = 4 - quarter_length
    return measure


def _has_rest_duration(
    tokens: list[Token],
    duration: Fraction,
    *,
    token_vocabulary: TokenVocabulary,
) -> bool:
    return any(
        isinstance(token, RestToken)
        and token_vocabulary.duration_vocabulary.id_to_fraction(token.duration_id) == duration
        for token in tokens
    )
