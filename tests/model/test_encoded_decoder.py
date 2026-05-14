from pathlib import Path

from musak_model.data.schema import SegmentMetadata
from musak_model.decoder import encoded_exercise_to_segment
from musak_model.tokens.schema import EndToken, Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def test_encoded_exercise_to_segment_decodes_token_ids(token_vocabulary: TokenVocabulary) -> None:
    tokens = [
        HandToken(hand=Hand.RIGHT),
        NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0),
        EndToken(),
    ]
    sample = EncodedExercise(
        token_ids=token_vocabulary.encode(tokens),
        bar_positions=[0, 0, 0],
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

    segment = encoded_exercise_to_segment(sample, token_vocabulary=token_vocabulary)

    assert segment.tokens == tokens
    assert segment.metadata == sample.metadata
