from collections import Counter
from fractions import Fraction

from numpy.random import default_rng

from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry
from musak_model.synthetic.substitution import (
    FigureByChordModel,
    SubstitutionConfig,
    chord_figure_log_probabilities,
    sample_substituted_figure,
)
from musak_model.tokens.schema import Hand, ScaleType

_TONIC = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)


def _figure(positions: list[int]) -> FigureNGram:
    return FigureNGram(onsets=tuple((((position, 0),), Fraction(1)) for position in positions))


def _entries(figures: list[FigureNGram]) -> tuple[FigureVocabularyEntry, ...]:
    vocabulary = FigureVocabulary.from_counts(
        {ScaleType.MAJOR: {Hand.RIGHT: {2: Counter({figure: 1 for figure in figures})}}}
    )
    return vocabulary.filter(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, n=2).entries


def _config(lambda_chord_figure: float) -> SubstitutionConfig:
    return SubstitutionConfig(
        lambda_curve=0.0,
        lambda_harm=0.0,
        lambda_accent=0.0,
        lambda_chord_figure=lambda_chord_figure,
        commonness_bias=0.0,
        max_resample_retries=4,
        monophonic=False,
    )


def test_table_returns_none_for_absent_chord_or_key() -> None:
    figure = _figure([0, 2])
    model = FigureByChordModel(log_probabilities={(ScaleType.MAJOR, Hand.RIGHT, 2, _TONIC): {figure: -0.5}})

    assert model.table(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2, chord=None) is None
    assert model.table(scale_type=ScaleType.MAJOR, hand=Hand.LEFT, figure_length=2, chord=_TONIC) is None
    assert model.table(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2, chord=_TONIC) == {figure: -0.5}


def test_chord_figure_log_probabilities_are_zero_without_a_table() -> None:
    figures = [_figure([0, 2]), _figure([0, -2])]

    assert chord_figure_log_probabilities(figures=figures, table=None) == [0.0, 0.0]


def test_chord_figure_log_probabilities_floor_unobserved_figures() -> None:
    seen = _figure([0, 2])
    unseen = _figure([0, -2])
    table = {seen: -0.25, _figure([0, 1]): -3.0}

    scores = chord_figure_log_probabilities(figures=[seen, unseen], table=table)

    assert scores[0] == -0.25
    assert scores[1] == -3.0  # the unobserved figure backs off to the least-likely observed log-probability


def test_empty_model_is_a_no_op_regardless_of_lambda() -> None:
    entries = _entries([_figure([0, 2]), _figure([0, -2])])
    kwargs = {
        "entries": entries,
        "anchor": 0,
        "target_slope": 0,
        "scale_type": ScaleType.MAJOR,
        "chord_pitch_classes": frozenset({0, 4, 7}),
        "envelope_value": 0.0,
        "metrical_position": 0,
        "grid_count_per_bar": 1,
        "chord": _TONIC,
    }

    with_term = sample_substituted_figure(config=_config(5.0), rng=default_rng(7), **kwargs)
    without_term = sample_substituted_figure(config=_config(0.0), rng=default_rng(7), **kwargs)

    assert with_term == without_term


def test_high_lambda_chord_figure_selects_the_chord_preferred_figure() -> None:
    preferred = _figure([0, -2])
    other = _figure([0, 2])
    entries = _entries([other, preferred])
    model = FigureByChordModel(
        log_probabilities={(ScaleType.MAJOR, Hand.RIGHT, 2, _TONIC): {preferred: 0.0, other: -10.0}}
    )

    chosen = sample_substituted_figure(
        entries=entries,
        anchor=0,
        target_slope=0,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=frozenset({0, 4, 7}),
        envelope_value=0.0,
        metrical_position=0,
        grid_count_per_bar=1,
        config=_config(50.0),
        rng=default_rng(0),
        chord=_TONIC,
        figure_by_chord_model=model,
    )

    assert chosen.figure == preferred
