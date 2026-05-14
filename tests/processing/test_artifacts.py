from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import SegmentMetadata
from musak_model.processing.ids import segment_id, source_id
from musak_model.processing.io import append_jsonl, load_encoded_jsonl, load_parsed_score_json, write_json_model
from musak_model.processing.snapshot import SpecialTokenSnapshotField, build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise
from musak_model.version import TOKENIZER_SCHEMA_VERSION
from tests.data.fixtures import bar, note_event, parsed_score


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


def test_parsed_score_json_round_trip(tmp_path: Path) -> None:
    score = parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([])],
    )
    path = tmp_path / "score.json"

    write_json_model(score, path, overwrite=True)
    loaded = load_parsed_score_json(path)

    assert loaded == score


def test_encoded_jsonl_round_trip(tmp_path: Path) -> None:
    metadata = SegmentMetadata(
        key_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=1,
        window_start_bar=0,
        source_file=Path("piece.mxl"),
    )
    sample = EncodedExercise(token_ids=[1, 2], bar_positions=[0, 0], metadata=metadata)
    path = tmp_path / "encoded.jsonl"

    line_index = append_jsonl(sample, path)

    assert line_index == 0
    assert load_encoded_jsonl(path) == [sample]


def test_tokenizer_snapshot_hash_changes_with_tokenization_config(tokenization_config: TokenizationConfig) -> None:
    second_config = TokenizationConfig(shortest_duration=32, allowed_tuplets=(3,), max_dots=1)

    first_snapshot = _snapshot(tokenization_config)
    second_snapshot = _snapshot(second_config)

    assert first_snapshot.tokenizer_hash != second_snapshot.tokenizer_hash
    assert first_snapshot.schema_version == TOKENIZER_SCHEMA_VERSION
    assert first_snapshot.vocabulary_size > 0
    assert first_snapshot.special_token_ids[SpecialTokenSnapshotField.BAR] < first_snapshot.vocabulary_size


def _snapshot(config: TokenizationConfig):
    duration_vocabulary = DurationVocabulary(config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    return build_tokenizer_snapshot(
        config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
