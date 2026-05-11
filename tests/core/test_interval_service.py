from unittest.mock import patch

from core.intervals.schema import IntervalRequest, IntervalResponse
from core.intervals.service import IntervalService


def test_get_config_returns_groups() -> None:
    response = IntervalService().get_config()
    assert response.groups


def test_generate_returns_interval_response() -> None:
    with (
        patch(
            "core.intervals.service.create_directory",
            return_value=("abc123", "/tmp/abc123"),
        ),
        patch("core.intervals.service.to_abjad"),
        patch("core.intervals.service.IntervalService._write_interval_info"),
        patch("core.intervals.service.Exporter") as mock_exporter,
    ):
        mock_exporter.return_value.export.return_value = (
            "/tmp/abc123/score.png",
            "/tmp/abc123/score.midi",
            "/tmp/abc123/score.mp3",
        )
        request = IntervalRequest(intervals={"perfect_fifth": 7})
        response = IntervalService().generate(request)

    assert isinstance(response, IntervalResponse)
    assert response.directory == "abc123"
    assert response.intervals == {"perfect_fifth": 7}


def test_generate_uses_all_config_intervals_when_none_requested() -> None:
    with (
        patch(
            "core.intervals.service.create_directory", return_value=("xyz", "/tmp/xyz")
        ),
        patch("core.intervals.service.to_abjad"),
        patch("core.intervals.service.IntervalService._write_interval_info"),
        patch("core.intervals.service.Exporter") as mock_exporter,
    ):
        mock_exporter.return_value.export.return_value = (
            "/tmp/xyz/score.png",
            "/tmp/xyz/score.midi",
            "/tmp/xyz/score.mp3",
        )
        response = IntervalService().generate(IntervalRequest())

    assert response.intervals
