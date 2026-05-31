import pytest

from musak_shared.misc import is_power_of_two, prime_factors


def test_is_power_of_two() -> None:
    assert is_power_of_two(1)
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(12)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, ()),
        (2, (2,)),
        (3, (3,)),
        (4, (2, 2)),
        (6, (2, 3)),
        (12, (2, 2, 3)),
        (9, (3, 3)),
        (5, (5,)),
        (60, (2, 2, 3, 5)),
    ],
)
def test_prime_factors(value: int, expected: tuple[int, ...]) -> None:
    assert prime_factors(value) == expected


def test_prime_factors_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        prime_factors(0)
