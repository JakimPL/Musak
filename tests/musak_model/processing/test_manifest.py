from fractions import Fraction
from pathlib import Path

from musak_model.data.config import SegmentationMode
from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.processing.manifest import EncodedManifestField, encoded_row
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType


def test_encoded_row_includes_segment_diagnostics(duration_vocabulary: DurationVocabulary, tmp_path: Path) -> None:
    source_path = tmp_path / "dataset" / "score.mxl"
    parsed_path = tmp_path / "processed" / "parsed" / "score.json"
    encoded_shard = tmp_path / "processed" / "encoded" / "abc" / "data-00000.jsonl"
    source_path.parent.mkdir()
    parsed_path.parent.mkdir(parents=True)
    encoded_shard.parent.mkdir(parents=True)
    source_path.touch()
    parsed_path.touch()
    encoded_shard.touch()
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=half_id),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("score.mxl"),
        ),
    )

    row = encoded_row(
        source_id_value="source",
        source_path=source_path,
        dataset_root=tmp_path / "dataset",
        parsed_path=parsed_path,
        processed_root=tmp_path / "processed",
        segment=segment,
        duration_vocabulary=duration_vocabulary,
        encoded_sample=None,
        encoded_shard=encoded_shard,
        encoded_line=None,
        segmentation_mode=SegmentationMode.WINDOWED,
    )

    assert row[EncodedManifestField.RIGHT_SILENCE_FRACTION] == 0.5
    assert row[EncodedManifestField.LEFT_SILENCE_FRACTION] == 1.0
    assert row[EncodedManifestField.ONE_HAND_ONLY] is True
    assert row[EncodedManifestField.DECLARED_KEY_FIFTHS] == ""
    assert row[EncodedManifestField.SILENT_BAR_COUNT] == 0
    assert row[EncodedManifestField.SILENT_BAR_FRACTION] == 0.0
    assert row[EncodedManifestField.SILENT_EDGE_BAR_COUNT] == 0
    assert row[EncodedManifestField.NOTE_TOKEN_FRACTION] == 0.5
