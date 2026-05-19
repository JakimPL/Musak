import pytest
from pydantic import ValidationError

from musak_shared.notation.schema import NoteData, ScoreData, StaveData


class TestNoteData:
    def test_rejects_invalid_duration(self) -> None:
        with pytest.raises(ValidationError):
            NoteData(keys=["c/4"], duration="xyz")

    def test_rejects_negative_dots(self) -> None:
        with pytest.raises(ValidationError):
            NoteData(keys=["c/4"], duration="w", dots=-1)

    def test_rejects_too_many_dots(self) -> None:
        with pytest.raises(ValidationError):
            NoteData(keys=["c/4"], duration="w", dots=3)

    def test_accepts_rest_duration(self) -> None:
        note = NoteData(keys=[], duration="wr")
        assert note.duration == "wr"
        assert note.keys == []


class TestStaveData:
    def test_rejects_invalid_clef(self) -> None:
        with pytest.raises(ValidationError):
            StaveData(clef="unknown", voices=[])

    def test_rejects_invalid_time_signature(self) -> None:
        with pytest.raises(ValidationError):
            StaveData(clef="treble", time_signature=(4, 3), voices=[])

    def test_accepts_none_time_signature(self) -> None:
        stave = StaveData(clef="treble", time_signature=None, voices=[])
        assert stave.time_signature is None

    def test_accepts_key_signature(self) -> None:
        stave = StaveData(clef="treble", key_signature="D", voices=[])
        assert stave.key_signature == "D"


class TestScoreData:
    def test_rejects_non_positive_tempo(self) -> None:
        with pytest.raises(ValidationError):
            ScoreData(rows=[], tempo=0)

        with pytest.raises(ValidationError):
            ScoreData(rows=[], tempo=-1)

    def test_accepts_none_tempo(self) -> None:
        score = ScoreData(rows=[], tempo=None)
        assert score.tempo is None
