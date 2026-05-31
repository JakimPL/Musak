import pytest

from musak_shared.misc import congruent_at_or_above, congruent_at_or_below, is_power_of_two, prime_factors


@pytest.mark.parametrize(
    ("value", "residue", "modulus", "expected"),
    [
        (10, 1, 7, 8),  # greatest <= 10 with x % 7 == 1
        (7, 1, 7, 1),  # 8 would exceed 7
        (8, 1, 7, 8),  # already congruent
        (0, 3, 7, -4),
    ],
)
def test_congruent_at_or_below(value: int, residue: int, modulus: int, expected: int) -> None:
    result = congruent_at_or_below(value, residue, modulus)

    assert result == expected
    assert result <= value
    assert result % modulus == residue % modulus


@pytest.mark.parametrize(
    ("value", "residue", "modulus", "expected"),
    [
        (2, 5, 7, 5),  # least >= 2 with x % 7 == 5
        (6, 5, 7, 12),  # 5 would be below 6
        (5, 5, 7, 5),  # already congruent
    ],
)
def test_congruent_at_or_above(value: int, residue: int, modulus: int, expected: int) -> None:
    result = congruent_at_or_above(value, residue, modulus)

    assert result == expected
    assert result >= value
    assert result % modulus == residue % modulus


def test_congruent_helpers_reject_non_positive_modulus() -> None:
    with pytest.raises(ValueError, match="modulus"):
        congruent_at_or_below(5, 1, 0)

    with pytest.raises(ValueError, match="modulus"):
        congruent_at_or_above(5, 1, 0)


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
