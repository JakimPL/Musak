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
    return func(integers, lambda x, y: x * y // math.gcd(x, y))


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def check_type(obj: object, obj_type: type) -> None:
    if not isinstance(obj, obj_type):
        raise TypeError(f"expected {obj_type.__name__}, got {type(obj).__name__}")
