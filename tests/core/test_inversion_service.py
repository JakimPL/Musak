from unittest.mock import patch

from core.inversions.schema import InversionRequest, InversionResponse
from core.inversions.service import InversionService


def test_get_config_returns_groups() -> None:
    response = InversionService().get_config()
    assert response.groups


def test_generate_returns_inversion_response() -> None:
    with (
        patch(
            "core.inversions.service.create_directory",
            return_value=("abc123", "/tmp/abc123"),
        ),
        patch("core.inversions.service.to_abjad"),
        patch("core.inversions.service.InversionService._write_chord_info"),
        patch("core.inversions.service.Exporter") as mock_exporter,
    ):
        mock_exporter.return_value.export.return_value = (
            "/tmp/abc123/score.png",
            "/tmp/abc123/score.midi",
            "/tmp/abc123/score.mp3",
        )
        response = InversionService().generate(InversionRequest())

    assert isinstance(response, InversionResponse)
    assert response.directory == "abc123"
    assert response.chord_types


def test_generate_uses_only_requested_chords() -> None:
    with (
        patch("core.inversions.service.create_directory", return_value=("xyz", "/tmp/xyz")),
        patch("core.inversions.service.to_abjad"),
        patch("core.inversions.service.InversionService._write_chord_info"),
        patch("core.inversions.service.Exporter") as mock_exporter,
    ):
        mock_exporter.return_value.export.return_value = (
            "/tmp/xyz/score.png",
            "/tmp/xyz/score.midi",
            "/tmp/xyz/score.mp3",
        )
        response = InversionService().generate(InversionRequest(chords={"": [0, 4, 7]}))

    assert response.chord_types == [""]
