import pathlib
from unittest.mock import patch

from musak.core.intervals.schema import IntervalRequest, IntervalResponse
from musak.core.intervals.service import IntervalService
from musak.core.notation.schema import ScoreData


def test_get_config_returns_groups() -> None:
    response = IntervalService().get_config()
    assert response.groups


def test_generate_returns_interval_response() -> None:
    with (
        patch(
            "musak.core.intervals.service.create_directory",
            return_value=("abc123", pathlib.Path("/tmp/abc123")),
        ),
        patch("musak.core.intervals.service.save_midi", return_value=pathlib.Path("/tmp/abc123/interval.mid")),
        patch("musak.core.intervals.service.IntervalService._write_interval_info"),
        patch("musak.core.intervals.service.Exporter"),
    ):
        request = IntervalRequest(intervals={"perfect_fifth": 7})
        response = IntervalService().generate(request)

    assert isinstance(response, IntervalResponse)
    assert response.directory == "abc123"
    assert response.intervals == {"perfect_fifth": 7}
    assert isinstance(response.score_data, ScoreData)


def test_generate_uses_all_config_intervals_when_none_requested() -> None:
    with (
        patch(
            "musak.core.intervals.service.create_directory",
            return_value=("xyz", pathlib.Path("/tmp/xyz")),
        ),
        patch("musak.core.intervals.service.save_midi", return_value=pathlib.Path("/tmp/xyz/interval.mid")),
        patch("musak.core.intervals.service.IntervalService._write_interval_info"),
        patch("musak.core.intervals.service.Exporter"),
    ):
        response = IntervalService().generate(IntervalRequest())

    assert response.intervals
