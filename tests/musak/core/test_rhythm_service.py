from unittest.mock import patch

from musak.core.rhythm.schema import RhythmRequest, RhythmResponse
from musak.core.rhythm.service import RhythmService
from musak_shared.notation.schema import ScoreData


def test_get_config_returns_groups() -> None:
    response = RhythmService().get_config()
    assert response.groups


def test_generate_returns_rhythm_response() -> None:
    with patch(
        "musak.core.rhythm.service.midi_to_audio",
        return_value="data:audio/mpeg;base64,abc",
    ):
        response = RhythmService().generate(RhythmRequest())

    assert isinstance(response, RhythmResponse)
    assert response.audio_data == "data:audio/mpeg;base64,abc"
    assert response.exception is None
    assert isinstance(response.score_data, ScoreData)


def test_generate_returns_exception_on_invalid_phrase_set() -> None:
    # notes=[2] is a half note (duration 1/2); gcd(1/2, 3/4)=1/4 but min_length=1/2 > 1/4
    # so the generator cannot tile a 3/4 measure and must raise InvalidPhraseSetError
    response = RhythmService().generate(RhythmRequest(notes=[2], phrases=[], time_signature=(3, 4)))
    assert response.exception is not None


def test_generate_sets_time_signature_error_for_non_power_of_two() -> None:
    with patch(
        "musak.core.rhythm.service.midi_to_audio",
        return_value="data:audio/mpeg;base64,abc",
    ):
        response = RhythmService().generate(RhythmRequest(time_signature=(4, 3)))

    assert response.time_signature_error is True
