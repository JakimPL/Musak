from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from musak_model.processing.ids import tokenizer_hash
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    HAND_HOME_OCTAVES,
    MAX_ACCIDENTAL,
    MAX_DEGREE,
    MAX_OCTAVE_OFFSET,
    MIN_ACCIDENTAL,
    MIN_DEGREE,
    MIN_DURATION_ID,
    MIN_OCTAVE_OFFSET,
    Hand,
)
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.version import TOKENIZER_SCHEMA_VERSION, SchemaVersion
from musak_shared.ratios import format_ratio


class TokenizerSnapshotField(StrEnum):
    SCHEMA_VERSION = "schema_version"
    TOKENIZATION_CONFIG = "tokenization_config"
    DURATION_FRACTIONS = "duration_fractions"
    VOCABULARY_SIZE = "vocabulary_size"
    SPECIAL_TOKEN_IDS = "special_token_ids"
    TOKEN_RANGES = "token_ranges"
    HAND_HOME_OCTAVES = "hand_home_octaves"


class SpecialTokenSnapshotField(StrEnum):
    START = "start"
    BAR = "bar"
    END = "end"
    RIGHT_HAND = "right_hand"
    LEFT_HAND = "left_hand"
    JOIN_WITH_PREVIOUS = "join_with_previous"
    FIRST_HOLD = "first_hold"


class TokenRangeSnapshotField(StrEnum):
    MIN_DEGREE = "min_degree"
    MAX_DEGREE = "max_degree"
    MIN_ACCIDENTAL = "min_accidental"
    MAX_ACCIDENTAL = "max_accidental"
    MIN_OCTAVE_OFFSET = "min_octave_offset"
    MAX_OCTAVE_OFFSET = "max_octave_offset"
    MIN_DURATION_ID = "min_duration_id"


TOKENIZER_SNAPSHOT_HASH_FIELDS: Final[tuple[TokenizerSnapshotField, ...]] = (
    TokenizerSnapshotField.SCHEMA_VERSION,
    TokenizerSnapshotField.TOKENIZATION_CONFIG,
    TokenizerSnapshotField.DURATION_FRACTIONS,
    TokenizerSnapshotField.VOCABULARY_SIZE,
    TokenizerSnapshotField.SPECIAL_TOKEN_IDS,
    TokenizerSnapshotField.TOKEN_RANGES,
    TokenizerSnapshotField.HAND_HOME_OCTAVES,
)


class TokenizerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion
    tokenizer_hash: str
    tokenization_config: dict[str, Any]
    duration_fractions: list[str]
    vocabulary_size: int = Field(ge=1)
    special_token_ids: dict[SpecialTokenSnapshotField, int]
    token_ranges: dict[TokenRangeSnapshotField, int]
    hand_home_octaves: dict[str, int]


def build_tokenizer_snapshot(
    tokenization_config: TokenizationConfig,
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> TokenizerSnapshot:
    tokenization_payload = tokenization_config.model_dump(mode="json")
    duration_fractions = [format_ratio(duration) for duration in duration_vocabulary.all_fractions()]
    special_token_ids = {
        SpecialTokenSnapshotField.START: token_vocabulary.start_token_id,
        SpecialTokenSnapshotField.BAR: token_vocabulary.bar_token_id,
        SpecialTokenSnapshotField.END: token_vocabulary.end_token_id,
        SpecialTokenSnapshotField.RIGHT_HAND: token_vocabulary.right_hand_token_id,
        SpecialTokenSnapshotField.LEFT_HAND: token_vocabulary.left_hand_token_id,
        SpecialTokenSnapshotField.JOIN_WITH_PREVIOUS: token_vocabulary.join_with_previous_token_id,
        SpecialTokenSnapshotField.FIRST_HOLD: token_vocabulary.first_hold_token_id,
    }
    token_ranges = {
        TokenRangeSnapshotField.MIN_DEGREE: MIN_DEGREE,
        TokenRangeSnapshotField.MAX_DEGREE: MAX_DEGREE,
        TokenRangeSnapshotField.MIN_ACCIDENTAL: MIN_ACCIDENTAL,
        TokenRangeSnapshotField.MAX_ACCIDENTAL: MAX_ACCIDENTAL,
        TokenRangeSnapshotField.MIN_OCTAVE_OFFSET: MIN_OCTAVE_OFFSET,
        TokenRangeSnapshotField.MAX_OCTAVE_OFFSET: MAX_OCTAVE_OFFSET,
        TokenRangeSnapshotField.MIN_DURATION_ID: MIN_DURATION_ID,
    }
    hand_home_octaves = {
        Hand.RIGHT.value: HAND_HOME_OCTAVES[Hand.RIGHT],
        Hand.LEFT.value: HAND_HOME_OCTAVES[Hand.LEFT],
    }
    hash_payload_values = {
        TokenizerSnapshotField.SCHEMA_VERSION: TOKENIZER_SCHEMA_VERSION,
        TokenizerSnapshotField.TOKENIZATION_CONFIG: tokenization_payload,
        TokenizerSnapshotField.DURATION_FRACTIONS: duration_fractions,
        TokenizerSnapshotField.VOCABULARY_SIZE: token_vocabulary.vocabulary_size,
        TokenizerSnapshotField.SPECIAL_TOKEN_IDS: special_token_ids,
        TokenizerSnapshotField.TOKEN_RANGES: token_ranges,
        TokenizerSnapshotField.HAND_HOME_OCTAVES: hand_home_octaves,
    }
    hash_payload = {field.value: hash_payload_values[field] for field in TOKENIZER_SNAPSHOT_HASH_FIELDS}
    return TokenizerSnapshot(
        schema_version=TOKENIZER_SCHEMA_VERSION,
        tokenizer_hash=tokenizer_hash(hash_payload),
        tokenization_config=tokenization_payload,
        duration_fractions=duration_fractions,
        vocabulary_size=token_vocabulary.vocabulary_size,
        special_token_ids=special_token_ids,
        token_ranges=token_ranges,
        hand_home_octaves=hand_home_octaves,
    )
