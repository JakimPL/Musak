import csv
import json
from pathlib import Path

from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.processing.diagnostic_report import write_dataset_diagnostic_report
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, RestToken
from musak_model.tokens.vocabulary import TokenVocabulary


def test_write_dataset_diagnostic_report_outputs_summary_and_tables(tmp_path: Path) -> None:
    token_vocabulary = _token_vocabulary()
    primary_processed = _write_processed_dataset(
        tmp_path,
        dataset_name="PDMX",
        encoded_hash="abc",
        token_vocabulary=token_vocabulary,
        token_counts=(8, 3),
    )
    reference_processed = _write_processed_dataset(
        tmp_path,
        dataset_name="exercises",
        encoded_hash="abc",
        token_vocabulary=token_vocabulary,
        token_counts=(3,),
    )

    result = write_dataset_diagnostic_report(
        dataset_name="PDMX",
        processed_directory=primary_processed,
        encoded_directory=primary_processed / "encoded" / "abc",
        output_directory=tmp_path / "diagnostics",
        scale_matcher_config=_scale_matcher_config(),
        reference_dataset_name="exercises",
        reference_processed_directory=reference_processed,
        reference_encoded_directory=reference_processed / "encoded" / "abc",
        max_sequence_length=4,
        top_rows=3,
        mlflow_db_path=None,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.report_path.is_file()
    assert result.table_paths["encoded_numeric_summary"].is_file()
    assert result.table_paths["reference_comparison"].is_file()
    assert summary["parsed"]["files"] == 2
    assert summary["parsed"]["parse_success_rate"] == 0.5
    assert summary["encoded"]["segments"] == 2
    assert summary["encoded"]["over_max_sequence_length"] == 1
    assert summary["tokens"]["token_kind_distribution"]["note"] == 2
    assert summary["tokens"]["token_kind_distribution"]["rest"] == 1
    assert summary["tonal_probes"][0]["probe"] == "A natural minor"
    assert summary["tonal_probes"][0]["reference_pitch_maps_to_degree_1"] is False
    assert summary["reference"]["name"] == "exercises"


def _write_processed_dataset(
    root: Path,
    *,
    dataset_name: str,
    encoded_hash: str,
    token_vocabulary: TokenVocabulary,
    token_counts: tuple[int, ...],
) -> Path:
    processed_directory = root / "processed" / dataset_name
    encoded_directory = processed_directory / "encoded" / encoded_hash
    encoded_directory.mkdir(parents=True)
    _write_parsed_manifest(processed_directory / "parsed.csv")
    _write_encoded_manifest(encoded_directory / "encoded.csv", token_counts=token_counts)
    _write_tokenizer_snapshot(encoded_directory / "tokenizer.json", token_vocabulary=token_vocabulary)
    _write_encoded_jsonl(encoded_directory / "data-00000.jsonl", token_vocabulary=token_vocabulary)
    return processed_directory


def _write_parsed_manifest(path: Path) -> None:
    rows = [
        {
            "source_id": "a",
            "status": "success",
            "error_type": "",
            "time_signature": "4/4",
            "declared_key_fifths": "0",
        },
        {
            "source_id": "b",
            "status": "error",
            "error_type": "ValueError",
            "time_signature": "",
            "declared_key_fifths": "",
        },
    ]
    _write_csv(path, rows)


def _write_encoded_manifest(path: Path, *, token_counts: tuple[int, ...]) -> None:
    rows = []
    for index, token_count in enumerate(token_counts):
        rows.append(
            {
                "segment_id": f"seg-{index}",
                "source_path": f"{index}.mxl",
                "encoded_line": str(index),
                "window_start_bar": "0",
                "bar_count": "4",
                "token_count": str(token_count),
                "eligible_for_training": "True",
                "ineligibility_reasons": "",
                "scale_root": "0",
                "scale_type": "major",
                "declared_key_fifths": "0",
                "scale_match_in_scale_weight_fraction": "1.0",
                "scale_match_out_of_scale_weight_fraction": "0.0",
                "scale_match_best_margin": "0.05",
                "scale_match_support_candidate_count": "2",
                "scale_match_tied_best_candidate_count": str(index + 1),
                "scale_match_declared_match_used": "True",
                "scale_match_low_confidence": "False",
                "scale_match_ambiguous": "True" if index == 1 else "False",
                "scale_match_no_pitches": "False",
                "time_signature": "4/4",
                "accidental_note_fraction": "0.25" if index == 1 else "0.0",
                "in_scale_note_fraction": "0.75" if index == 1 else "1.0",
                "note_density_per_beat": "1.5",
                "onset_density_per_beat": "1.25",
                "right_onset_density_per_beat": "0.75",
                "left_onset_density_per_beat": "0.5",
                "shortest_note_duration_beats": "0.5",
                "both_hands_active_fraction": "0.5",
                "hand_activity_balance": "1.0",
                "max_notes_per_onset": "2",
                "max_notes_per_hand": "1",
                "max_onset_span_semitones": "5",
                "max_melodic_gap_semitones": "7",
                "static_hand_span_degrees": "6",
                "synchronized_onset_fraction": "0.25",
                "independent_onset_fraction": "0.75",
                "empty_score": "False",
                "one_hand_only": "False",
                "has_dotted_notes": "False",
            }
        )
    _write_csv(path, rows)


def _write_tokenizer_snapshot(path: Path, *, token_vocabulary: TokenVocabulary) -> None:
    tokenization_config = TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=token_vocabulary.duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    path.write_text(snapshot.model_dump_json(), encoding="utf-8")


def _write_encoded_jsonl(path: Path, *, token_vocabulary: TokenVocabulary) -> None:
    token_ids = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=3),
            RestToken(duration_id=3),
            BarToken(),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=3),
            EndToken(),
        ]
    )
    row = {"token_ids": token_ids, "bar_positions": [0 for _ in token_ids]}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _token_vocabulary() -> TokenVocabulary:
    tokenization_config = TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)
    return TokenVocabulary(DurationVocabulary(tokenization_config))


def _scale_matcher_config() -> ScaleMatcherConfig:
    return ScaleMatcherConfig(
        support_score_margin=0.08,
        selection_score_margin=0.03,
        maximum_unexplained_weight_fraction=0.10,
        maximum_explanation_pitch_class_count=9,
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
