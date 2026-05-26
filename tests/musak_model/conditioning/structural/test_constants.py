from musak_model.conditioning.structural.constants import FALSE_CONTROL_ID, TRUE_CONTROL_ID


def test_boolean_control_ids_are_stable() -> None:
    assert FALSE_CONTROL_ID == 1
    assert TRUE_CONTROL_ID == 2
