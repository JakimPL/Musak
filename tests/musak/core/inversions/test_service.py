from unittest.mock import patch

from musak.core.inversions.schema import InversionRequest, InversionResponse
from musak.core.inversions.service import InversionService
from musak_shared.notation.schema import ScoreData


def test_get_config_returns_groups() -> None:
    response = InversionService().get_config()
    assert response.groups


def test_generate_returns_inversion_response() -> None:
    with patch(
        "musak.core.inversions.service.midi_to_audio",
        return_value="data:audio/mpeg;base64,abc",
    ):
        response = InversionService().generate(InversionRequest())

    assert isinstance(response, InversionResponse)
    assert response.audio_data == "data:audio/mpeg;base64,abc"
    assert response.chord_types
    assert isinstance(response.score_data, ScoreData)


def test_generate_uses_only_requested_chords() -> None:
    with patch(
        "musak.core.inversions.service.midi_to_audio",
        return_value="data:audio/mpeg;base64,abc",
    ):
        response = InversionService().generate(InversionRequest(chords={"m": [0, 3, 7]}))

    assert response.chord_types == ["m"]
