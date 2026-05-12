import pathlib
from unittest.mock import patch

from musak.core.notation.schema import ScoreData
from musak.core.rhythm.schema import RhythmRequest, RhythmResponse
from musak.core.rhythm.service import RhythmService


def test_get_config_returns_groups() -> None:
    response = RhythmService().get_config()
    assert response.groups


def test_generate_returns_rhythm_response() -> None:
    with (
        patch(
            "musak.core.rhythm.service.create_directory",
            return_value=("abc123", pathlib.Path("/tmp/abc123")),
        ),
        patch(
            "musak.core.rhythm.service.save_rhythm_midi",
            return_value=pathlib.Path("/tmp/abc123/rhythm.mid"),
        ),
        patch("musak.core.rhythm.service.Exporter") as mock_exporter,
    ):
        mock_exporter.return_value.export_audio.return_value = pathlib.Path("/tmp/abc123/rhythm.mp3")
        response = RhythmService().generate(RhythmRequest())

    assert isinstance(response, RhythmResponse)
    assert response.directory == "abc123"
    assert response.exception is None
    assert isinstance(response.score_data, ScoreData)


def test_generate_returns_exception_on_invalid_phrase_set() -> None:
    # notes=[2] is a half note (duration 1/2); gcd(1/2, 3/4)=1/4 but min_length=1/2 > 1/4
    # so the generator cannot tile a 3/4 measure and must raise InvalidPhraseSetError
    response = RhythmService().generate(RhythmRequest(notes=[2], phrases=[], time_signature=(3, 4)))
    assert response.exception is not None


def test_generate_sets_time_signature_error_for_non_power_of_two() -> None:
    with (
        patch(
            "musak.core.rhythm.service.create_directory",
            return_value=("xyz", pathlib.Path("/tmp/xyz")),
        ),
        patch(
            "musak.core.rhythm.service.save_rhythm_midi",
            return_value=pathlib.Path("/tmp/xyz/rhythm.mid"),
        ),
        patch("musak.core.rhythm.service.Exporter") as mock_exporter,
    ):
        mock_exporter.return_value.export_audio.return_value = pathlib.Path("/tmp/xyz/rhythm.mp3")
        response = RhythmService().generate(RhythmRequest(time_signature=(4, 3)))

    assert response.time_signature_error is True
