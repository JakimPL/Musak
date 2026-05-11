from unittest.mock import patch

from core.rhythm.schema import RhythmRequest, RhythmResponse
from core.rhythm.service import RhythmService


def test_get_config_returns_groups() -> None:
    response = RhythmService().get_config()
    assert response.groups


def test_generate_returns_rhythm_response() -> None:
    with (
        patch(
            "core.rhythm.service.create_directory",
            return_value=("abc123", "/tmp/abc123"),
        ),
        patch("core.rhythm.service.Exporter") as mock_exporter,
    ):
        mock_exporter.return_value.export.return_value = (
            "/tmp/abc123/score.png",
            "/tmp/abc123/score.midi",
            "/tmp/abc123/score.mp3",
        )
        response = RhythmService().generate(RhythmRequest())

    assert isinstance(response, RhythmResponse)
    assert response.directory == "abc123"
    assert response.exception is None


def test_generate_returns_exception_on_invalid_phrase_set() -> None:
    # notes=[2] is a half note (duration 1/2); gcd(1/2, 3/4)=1/4 but min_length=1/2 > 1/4
    # so the generator cannot tile a 3/4 measure and must raise InvalidPhraseSetError
    response = RhythmService().generate(RhythmRequest(notes=[2], phrases=[], time_signature=(3, 4)))
    assert response.exception is not None


def test_generate_sets_time_signature_error_for_non_power_of_two() -> None:
    with (
        patch("core.rhythm.service.create_directory", return_value=("xyz", "/tmp/xyz")),
        patch("core.rhythm.service.Exporter") as mock_exporter,
    ):
        mock_exporter.return_value.export.return_value = (
            "/tmp/xyz/score.png",
            "/tmp/xyz/score.midi",
            "/tmp/xyz/score.mp3",
        )
        response = RhythmService().generate(RhythmRequest(time_signature=(4, 3)))

    assert response.time_signature_error is True
