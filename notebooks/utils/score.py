from musak_model.data.schema import Segment
from musak_model.decoder import segment_to_music21_score
from musak_model.tokens.duration import DurationVocabulary


def score_summary(segment: Segment, *, duration_vocabulary: DurationVocabulary) -> list[dict[str, str | int]]:
    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    rows: list[dict[str, str | int]] = []
    for part in score.parts:
        flattened = part.flatten()
        rows.append(
            {
                "part": str(part.id),
                "notes_or_chords": len(flattened.notes),
                "highest_time": str(flattened.highestTime),
            }
        )

    return rows
