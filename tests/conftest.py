import pytest

from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary


@pytest.fixture
def tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)


@pytest.fixture
def duration_vocabulary(tokenization_config: TokenizationConfig) -> DurationVocabulary:
    return DurationVocabulary(tokenization_config)


@pytest.fixture
def token_vocabulary(duration_vocabulary: DurationVocabulary) -> TokenVocabulary:
    return TokenVocabulary(duration_vocabulary)
