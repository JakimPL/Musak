def is_power_of_two(value: int) -> bool:
    if value <= 0:
        return False

    while value != 1:
        if value % 2:
            return False
        value //= 2

    return True
