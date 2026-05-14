from __future__ import annotations

from musak_model.data.schema import Segment
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def encoded_exercise_to_segment(
    sample: EncodedExercise,
    *,
    token_vocabulary: TokenVocabulary,
) -> Segment:
    tokens = token_vocabulary.decode(sample.token_ids)
    return Segment(tokens=tokens, metadata=sample.metadata)
