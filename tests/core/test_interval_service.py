from unittest.mock import patch

from musak.core.intervals.schema import IntervalRequest, IntervalResponse
from musak.core.intervals.service import IntervalService
from musak.core.notation.schema import ScoreData


def test_get_config_returns_groups() -> None:
    response = IntervalService().get_config()
    assert response.groups


def test_generate_returns_interval_response() -> None:
    with patch(
        "musak.core.intervals.service.midi_to_audio",
        return_value="data:audio/mpeg;base64,abc",
    ):
        request = IntervalRequest(intervals={"perfect_fifth": 7})
        response = IntervalService().generate(request)

    assert isinstance(response, IntervalResponse)
    assert response.audio_data == "data:audio/mpeg;base64,abc"
    assert response.intervals == {"perfect_fifth": 7}
    assert isinstance(response.score_data, ScoreData)


def test_generate_uses_all_config_intervals_when_none_requested() -> None:
    with patch(
        "musak.core.intervals.service.midi_to_audio",
        return_value="data:audio/mpeg;base64,abc",
    ):
        response = IntervalService().generate(IntervalRequest())

    assert response.intervals
