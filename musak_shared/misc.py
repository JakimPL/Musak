import math
from typing import Callable, List


def func(integers: List[int], function: Callable[[int, int], int]) -> int:
    value = integers[0]
    for integer in integers:
        value = function(value, integer)

    return value


def gcd(integers: List[int]) -> int:
    return func(integers, math.gcd)


def lcm(integers: List[int]) -> int:
    return func(integers, math.lcm)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def congruent_at_or_below(value: int, residue: int, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")

    return value - ((value - residue) % modulus)


def congruent_at_or_above(value: int, residue: int, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")

    return value + ((residue - value) % modulus)


def prime_factors(value: int) -> tuple[int, ...]:
    if value < 1:
        raise ValueError("value must be positive")

    factors: list[int] = []
    remaining = value
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def check_type(obj: object, obj_type: type) -> None:
    if not isinstance(obj, obj_type):
        raise TypeError(f"expected {obj_type.__name__}, got {type(obj).__name__}")
