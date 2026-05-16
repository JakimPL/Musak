from pathlib import Path

from pydantic import BaseModel, ConfigDict

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.tokens.schema import Hand, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary


class EncodedExercise(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    token_ids: list[int]
    bar_positions: list[int]
    hand: Hand | None = None
    metadata: SegmentMetadata

    @property
    def source_file(self) -> Path:
        return self.metadata.source_file

    @property
    def key_root(self) -> int:
        return self.metadata.key_root

    @property
    def scale_type(self) -> ScaleType:
        return self.metadata.scale_type

    @property
    def time_numerator(self) -> int:
        return self.metadata.time_numerator

    @property
    def time_denominator(self) -> int:
        return self.metadata.time_denominator

    @property
    def difficulty_level(self) -> int | None:
        return self.metadata.difficulty_level

    def to_segment(self, *, token_vocabulary: TokenVocabulary) -> Segment:
        return Segment(tokens=token_vocabulary.decode(self.token_ids), metadata=self.metadata)


class IngestionErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    exception_type: str
    message: str


class IngestionSplit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train: list[EncodedExercise]
    validation: list[EncodedExercise]
    invalid_files: list[IngestionErrorRecord]
