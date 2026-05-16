from musak_shared.common import is_power_of_two


def test_is_power_of_two() -> None:
    assert is_power_of_two(1)
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(12)
