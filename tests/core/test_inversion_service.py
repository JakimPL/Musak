import pathlib
from unittest.mock import patch

from musak.core.inversions.schema import InversionRequest, InversionResponse
from musak.core.inversions.service import InversionService
from musak.core.notation.schema import ScoreData


def test_get_config_returns_groups() -> None:
    response = InversionService().get_config()
    assert response.groups


def test_generate_returns_inversion_response() -> None:
    with (
        patch(
            "musak.core.inversions.service.create_directory",
            return_value=("abc123", pathlib.Path("/tmp/abc123")),
        ),
        patch("musak.core.inversions.service.to_abjad"),
        patch("musak.core.inversions.service.InversionService._write_chord_info"),
        patch("musak.core.inversions.service.Exporter"),
    ):
        response = InversionService().generate(InversionRequest())

    assert isinstance(response, InversionResponse)
    assert response.directory == "abc123"
    assert response.chord_types
    assert isinstance(response.score_data, ScoreData)


def test_generate_uses_only_requested_chords() -> None:
    with (
        patch(
            "musak.core.inversions.service.create_directory",
            return_value=("xyz", pathlib.Path("/tmp/xyz")),
        ),
        patch("musak.core.inversions.service.to_abjad"),
        patch("musak.core.inversions.service.InversionService._write_chord_info"),
        patch("musak.core.inversions.service.Exporter"),
    ):
        response = InversionService().generate(InversionRequest(chords={"m": [0, 3, 7]}))

    assert response.chord_types == ["m"]
