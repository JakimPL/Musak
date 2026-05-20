from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.data.schema import ParsedBar, ParsedNote, ParsedScore, SegmentMetadata
from musak_model.processing.io import append_jsonl, write_json_model
from musak_model.processing.manifest import EncodedManifestField
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise
from notebooks.utils.encoded import load_encoded_manifest_selection


def test_load_encoded_manifest_selection_decodes_selected_manifest_row(tmp_path: Path) -> None:
    tokenization_config = TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    encoded_dir = tmp_path / "encoded" / "abc"
    encoded_shard = encoded_dir / "data-00000.jsonl"
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    write_json_model(snapshot, encoded_dir / "tokenizer.json", overwrite=True)
    duration_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    sample = EncodedExercise(
        token_ids=token_vocabulary.encode(
            [
                HandToken(hand=Hand.RIGHT),
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=duration_id),
            ]
        ),
        bar_positions=[0, 0],
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
        ),
    )
    encoded_line = append_jsonl(sample, encoded_shard)

    selection = load_encoded_manifest_selection(
        {
            EncodedManifestField.ENCODED_SHARD.value: "encoded/abc/data-00000.jsonl",
            EncodedManifestField.ENCODED_LINE.value: encoded_line,
        },
        dataset_dir=tmp_path,
    )

    assert selection.encoded_line == 0
    assert selection.segment.tokens == sample.to_segment(token_vocabulary=token_vocabulary).tokens
    assert selection.shard.path == encoded_shard


def test_load_encoded_manifest_selection_uses_encoded_dir_when_row_has_no_shard(tmp_path: Path) -> None:
    encoded_dir, sample, token_vocabulary = _write_encoded_run(tmp_path)

    selection = load_encoded_manifest_selection(
        {
            EncodedManifestField.ENCODED_LINE.value: 0,
        },
        dataset_dir=tmp_path,
        encoded_dir=encoded_dir,
    )

    assert selection.segment.tokens == sample.to_segment(token_vocabulary=token_vocabulary).tokens


def test_load_encoded_manifest_selection_rejects_rows_without_encoded_sample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="select an encoded run"):
        load_encoded_manifest_selection(
            {
                EncodedManifestField.ENCODED_SHARD.value: "encoded/abc/data-00000.jsonl",
                EncodedManifestField.ENCODED_LINE.value: "",
            },
            dataset_dir=tmp_path,
        )


def test_load_encoded_manifest_selection_reconstructs_ineligible_row_from_parsed_score(tmp_path: Path) -> None:
    encoded_dir, _, _ = _write_encoded_run(tmp_path)
    parsed_path = tmp_path / "parsed" / "score.json"
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            _note_bar(60),
            _note_bar(62),
        ],
        left_hand_bars=[
            _note_bar(48),
            _note_bar(50),
        ],
    )
    write_json_model(score, parsed_path, overwrite=True)

    selection = load_encoded_manifest_selection(
        {
            EncodedManifestField.PARSED_PATH.value: "parsed/score.json",
            EncodedManifestField.SOURCE_PATH.value: "score.mxl",
            EncodedManifestField.ENCODED_LINE.value: "",
            EncodedManifestField.WINDOW_START_BAR.value: 1,
            EncodedManifestField.BAR_COUNT.value: 1,
        },
        dataset_dir=tmp_path,
        encoded_dir=encoded_dir,
    )

    assert selection.encoded_line is None
    assert selection.shard is None
    assert selection.segment.metadata.window_start_bar == 1
    assert selection.segment.tokens


def test_load_encoded_manifest_selection_recovers_parsed_path_from_source_id(tmp_path: Path) -> None:
    encoded_dir, _, _ = _write_encoded_run(tmp_path)
    source_id = "abcdef"
    parsed_path = tmp_path / "parsed" / source_id[0] / f"{source_id}.json"
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[_note_bar(60)],
        left_hand_bars=[_note_bar(48)],
    )
    write_json_model(score, parsed_path, overwrite=True)

    selection = load_encoded_manifest_selection(
        {
            EncodedManifestField.SOURCE_ID.value: source_id,
            EncodedManifestField.SOURCE_PATH.value: "score.mxl",
            EncodedManifestField.ENCODED_LINE.value: "",
            EncodedManifestField.WINDOW_START_BAR.value: 0,
            EncodedManifestField.BAR_COUNT.value: 1,
        },
        dataset_dir=tmp_path,
        encoded_dir=encoded_dir,
    )

    assert selection.segment.tokens


def _note_bar(midi_pitch: int) -> ParsedBar:
    return ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[ParsedNote(midi_pitch=midi_pitch, duration=Fraction(1, 4), beat_offset=Fraction(0))],
    )


def _write_encoded_run(tmp_path: Path) -> tuple[Path, EncodedExercise, TokenVocabulary]:
    tokenization_config = TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    encoded_dir = tmp_path / "encoded" / "abc"
    encoded_shard = encoded_dir / "data-00000.jsonl"
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    write_json_model(snapshot, encoded_dir / "tokenizer.json", overwrite=True)
    duration_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    sample = EncodedExercise(
        token_ids=token_vocabulary.encode(
            [
                HandToken(hand=Hand.RIGHT),
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=duration_id),
            ]
        ),
        bar_positions=[0, 0],
        metadata=SegmentMetadata(
            key_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
        ),
    )
    append_jsonl(sample, encoded_shard)
    return encoded_dir, sample, token_vocabulary
