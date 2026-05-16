import pytest

from musak_model.common.validators import is_power_of_two


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, True),  # 2^0
        (2, True),  # 2^1
        (4, True),  # 2^2
        (8, True),  # 2^3
        (16, True),  # 2^4
        (0, False),  # Not positive
        (-2, False),  # Negative number
        (3, False),  # Not a power of two
        (6, False),  # Not a power of two
        (12, False),  # Not a power of two
    ],
)
def test_is_power_of_two(value, expected) -> None:
    assert is_power_of_two(value) == expected
