import pytest

from musak_shared.time_signature import validate_time_signature


class TestValidateTimeSignature:
    def test_accepts_valid(self) -> None:
        validate_time_signature((4, 4)) == (4, 4)
        validate_time_signature((3, 8)) == (3, 8)
        validate_time_signature((7, 16)) == (7, 16)

    def test_rejects_zero_numerator(self) -> None:
        with pytest.raises(ValueError):
            validate_time_signature((0, 4))

    def test_rejects_negative_numerator(self) -> None:
        with pytest.raises(ValueError):
            validate_time_signature((-1, 4))

    def test_rejects_non_power_of_two_denominator(self) -> None:
        with pytest.raises(ValueError):
            validate_time_signature((4, 3))

    def test_rejects_zero_denominator(self) -> None:
        with pytest.raises(ValueError):
            validate_time_signature((4, 0))
