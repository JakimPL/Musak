from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from math import log
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.harmony.schema import Chord
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.chord.schema import (
    INITIAL_CHORD_SOURCE,
    ChordTransitionCounts,
    FigureByChordCounts,
    chord_from_key,
)
from musak_model.paths import CHORD_FIT_CONFIG_PATH
from musak_model.synthetic.processes.chord_track import ChordTransitionModel
from musak_model.synthetic.substitution.chord_figure import FigureByChordKey, FigureByChordModel
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.files import load_yaml_config


class ChordFitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prior_count: float = Field(gt=0.0)
    functional_strength: float = Field(ge=0.0, le=1.0)
    self_transition_bias: float = Field(ge=0.0, le=1.0)

    @classmethod
    def load(cls, path: Path = CHORD_FIT_CONFIG_PATH) -> ChordFitConfig:
        return cls.model_validate(load_yaml_config(path))


def fit_chord_transition_model(
    transition_counts: ChordTransitionCounts,
    *,
    scale_type: ScaleType,
    prior: ChordTransitionModel,
    prior_count: float,
) -> ChordTransitionModel:
    if prior_count <= 0.0:
        raise ValueError("prior_count must be positive")

    chords = tuple(prior.initial_distribution.keys())
    chord_set = set(chords)
    initial_empirical, transition_empirical = _empirical_counts(
        transition_counts, scale_type=scale_type, chord_set=chord_set
    )
    initial_distribution = _smoothed_row(
        empirical=initial_empirical,
        prior_row=prior.initial_distribution,
        prior_count=prior_count,
        chords=chords,
    )
    transitions = {
        source: _smoothed_row(
            empirical=transition_empirical.get(source, {}),
            prior_row=prior.transitions[source],
            prior_count=prior_count,
            chords=chords,
        )
        for source in chords
    }
    return ChordTransitionModel(initial_distribution=initial_distribution, transitions=transitions)


def fit_figure_by_chord(figure_by_chord_counts: FigureByChordCounts) -> FigureByChordModel:
    grouped: dict[FigureByChordKey, dict[FigureNGram, float]] = defaultdict(dict)
    totals: dict[FigureByChordKey, float] = defaultdict(float)
    for key, count in figure_by_chord_counts.items():
        model_key = (ScaleType(key.scale_type), Hand(key.hand), key.figure_length, chord_from_key(key.chord))
        figure = FigureNGram.model_validate_json(key.figure)
        grouped[model_key][figure] = grouped[model_key].get(figure, 0.0) + count
        totals[model_key] += count

    log_probabilities: dict[FigureByChordKey, dict[FigureNGram, float]] = {
        model_key: {figure: log(count / totals[model_key]) for figure, count in figures.items()}
        for model_key, figures in grouped.items()
    }
    return FigureByChordModel(log_probabilities=log_probabilities)


def _empirical_counts(
    transition_counts: ChordTransitionCounts,
    *,
    scale_type: ScaleType,
    chord_set: set[Chord],
) -> tuple[dict[Chord, float], dict[Chord, dict[Chord, float]]]:
    initial: dict[Chord, float] = defaultdict(float)
    transitions: dict[Chord, dict[Chord, float]] = defaultdict(lambda: defaultdict(float))
    for key, count in transition_counts.items():
        if key.scale_type != scale_type.value:
            continue

        destination = chord_from_key(key.destination_chord)
        if destination not in chord_set:
            continue

        if key.source_chord == INITIAL_CHORD_SOURCE:
            initial[destination] += count
            continue

        source = chord_from_key(key.source_chord)
        if source in chord_set:
            transitions[source][destination] += count

    return initial, transitions


def _smoothed_row(
    *,
    empirical: Mapping[Chord, float],
    prior_row: Mapping[Chord, float],
    prior_count: float,
    chords: tuple[Chord, ...],
) -> dict[Chord, float]:
    denominator = sum(empirical.values()) + prior_count
    return {chord: (empirical.get(chord, 0.0) + prior_count * prior_row[chord]) / denominator for chord in chords}
