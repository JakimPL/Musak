from collections.abc import Iterable

import pandas as pd

from musak_model.data.schema import ParsedScore, Segment
from musak_model.decoder import PianoRollEvent, parsed_score_to_piano_roll_events, segment_to_piano_roll_events
from musak_model.tokens.duration import DurationVocabulary


def piano_roll_dataframe(segment: Segment, *, duration_vocabulary: DurationVocabulary) -> pd.DataFrame:
    return _events_to_dataframe(segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary))


def parsed_score_piano_roll_dataframe(score: ParsedScore) -> pd.DataFrame:
    return _events_to_dataframe(parsed_score_to_piano_roll_events(score))


def _events_to_dataframe(events: Iterable[PianoRollEvent]) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "hand": event.hand.value,
                "midi_pitch": event.midi_pitch,
                "start": float(event.start),
                "duration": float(event.duration),
                "end": float(event.end),
            }
        )

    return pd.DataFrame(rows)
