def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0
