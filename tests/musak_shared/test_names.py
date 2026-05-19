from musak_shared.names import midi_to_vexflow_key


class TestMidiToVexflowKey:
    def test_middle_c(self) -> None:
        assert midi_to_vexflow_key(60) == "c/4"

    def test_sharp(self) -> None:
        assert midi_to_vexflow_key(61) == "c#/4"

    def test_below_middle_c(self) -> None:
        assert midi_to_vexflow_key(59) == "b/3"

    def test_octave_boundary(self) -> None:
        assert midi_to_vexflow_key(48) == "c/3"
        assert midi_to_vexflow_key(72) == "c/5"
