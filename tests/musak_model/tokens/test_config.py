import pytest
from pydantic import ValidationError

from musak_model.tokens.config import TokenizationConfig


class TestTokenizationConfig:
    def test_requires_all_fields(self) -> None:
        with pytest.raises(ValidationError):
            TokenizationConfig.model_validate({})

    def test_accepts_empty_allowed_tuplets(self) -> None:
        config = TokenizationConfig(shortest_duration=16, allowed_tuplets=(), max_dots=1)

        assert config.allowed_tuplets == ()

    def test_rejects_tuplet_divisor_less_than_two(self) -> None:
        with pytest.raises(ValidationError, match="allowed tuplet divisors"):
            TokenizationConfig(shortest_duration=16, allowed_tuplets=(1,), max_dots=1)

    def test_rejects_non_power_of_two_shortest_duration(self) -> None:
        with pytest.raises(ValidationError, match="power of two"):
            TokenizationConfig(shortest_duration=12, allowed_tuplets=(), max_dots=1)
