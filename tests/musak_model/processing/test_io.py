from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.processing.io import append_jsonl, load_encoded_jsonl, load_parsed_score_json, write_json_model
from musak_model.tokens.schema import ScaleType
from musak_model.training.ingestion.schema import EncodedExercise
from tests.musak_model.data.fixtures import bar, note_event, parsed_score


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
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
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
