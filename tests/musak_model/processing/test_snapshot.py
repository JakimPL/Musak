from musak_model.processing.snapshot import SpecialTokenSnapshotField, TokenizerSnapshot, build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.version import TOKENIZER_SCHEMA_VERSION


def test_tokenizer_snapshot_hash_changes_with_tokenization_config(tokenization_config: TokenizationConfig) -> None:
    second_config = TokenizationConfig(shortest_duration=32, allowed_tuplets=(3,), max_dots=1)

    first_snapshot = _snapshot(tokenization_config)
    second_snapshot = _snapshot(second_config)

    assert first_snapshot.tokenizer_hash != second_snapshot.tokenizer_hash
    assert first_snapshot.schema_version == TOKENIZER_SCHEMA_VERSION
    assert first_snapshot.vocabulary_size > 0
    assert first_snapshot.special_token_ids[SpecialTokenSnapshotField.START] < first_snapshot.vocabulary_size
    assert first_snapshot.special_token_ids[SpecialTokenSnapshotField.BAR] < first_snapshot.vocabulary_size


def _snapshot(config: TokenizationConfig) -> TokenizerSnapshot:
    duration_vocabulary = DurationVocabulary(config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    return build_tokenizer_snapshot(
        config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
