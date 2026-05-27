from musak_shared.notation.html import score_data_html
from musak_shared.notation.schema import ScoreData, StaveData, VoiceData


def test_score_data_html_serializes_key_and_time_signatures() -> None:
    score = ScoreData(
        rows=[
            [
                StaveData(
                    clef="treble",
                    key_signature="D",
                    time_signature=(3, 4),
                    voices=[VoiceData(notes=[])],
                )
            ]
        ]
    )

    html = score_data_html(score)

    assert r"\"key_signature\": \"D\"" in html
    assert r"\"time_signature\": [3, 4]" in html
    assert r"\"layout\": \"separate_hand_rows\"" in html
