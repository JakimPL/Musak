from collections import Counter
from pathlib import Path

from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.n_grams.profile.chord.io import (
    read_chord_metadata,
    read_chord_transitions,
    read_figure_by_chord,
    write_chord_metadata,
    write_chord_transitions,
    write_figure_by_chord,
)
from musak_model.n_grams.profile.chord.schema import (
    INITIAL_CHORD_SOURCE,
    ChordProfileMetadata,
    ChordTransitionKey,
    FigureByChordCountKey,
    chord_to_key,
)

_TONIC = chord_to_key(Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR))
_DOMINANT = chord_to_key(Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR))


def test_chord_transitions_round_trip(tmp_path: Path) -> None:
    counts = Counter(
        {
            ChordTransitionKey("major", INITIAL_CHORD_SOURCE, _TONIC): 3,
            ChordTransitionKey("major", _TONIC, _DOMINANT): 5,
            ChordTransitionKey("major", _DOMINANT, _TONIC): 4,
            ChordTransitionKey("harmonic_minor", _DOMINANT, _TONIC): 2,
        }
    )
    path = tmp_path / "transitions.parquet"

    write_chord_transitions(counts, path)

    assert read_chord_transitions(path) == counts


def test_figure_by_chord_round_trip(tmp_path: Path) -> None:
    counts = Counter(
        {
            FigureByChordCountKey("major", "right", 2, _TONIC, '{"onsets":[[[[0,0]],"1"]]}'): 7,
            FigureByChordCountKey("major", "right", 2, _DOMINANT, '{"onsets":[[[[0,0]],"1"]]}'): 2,
        }
    )
    path = tmp_path / "figure_by_chord.parquet"

    write_figure_by_chord(counts, path)

    assert read_figure_by_chord(path) == counts


def test_chord_metadata_round_trip(tmp_path: Path) -> None:
    metadata = ChordProfileMetadata(resolution=1, self_transition_bias=0.25, non_chord_penalty=1.0, sample_count=42)
    path = tmp_path / "metadata.json"

    write_chord_metadata(metadata, path)

    assert read_chord_metadata(path) == metadata
