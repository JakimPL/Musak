from fractions import Fraction

from numpy.random import default_rng

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabularyEntry, FigureVocabularyGroup
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.figure_selection import figure_fits_slot, select_figure
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType

_TONIC_PITCH_CLASSES = frozenset({0, 4, 7})


def _ngram(*steps: int) -> FigureNGram:
    return FigureNGram(onsets=tuple((((step, 0),), Fraction(1)) for step in steps))


def _entry(figure: FigureNGram, count: int) -> FigureVocabularyEntry:
    group = FigureVocabularyGroup(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, n=len(figure.onsets))
    return FigureVocabularyEntry(group=group, figure=figure, count=count)


def _config(**overrides: float) -> RenderConfig:
    return RenderConfig.load().model_copy(update=overrides)


def _vocabulary() -> DurationVocabulary:
    return DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))


def test_figure_fits_slot_admits_single_note_and_rejects_too_fast_figures() -> None:
    vocabulary = _vocabulary()

    assert figure_fits_slot(
        _ngram(0), slot_duration=Fraction(1, 4), shortest_note_duration=Fraction(1, 16), duration_vocabulary=vocabulary
    )
    # Eight onsets across a quarter note would each be a 1/32 — below the 1/16 floor.
    assert not figure_fits_slot(
        _ngram(0, 1, 2, 3, 4, 5, 6, 7),
        slot_duration=Fraction(1, 4),
        shortest_note_duration=Fraction(1, 16),
        duration_vocabulary=vocabulary,
    )


def test_select_figure_is_count_proportional_at_zero_lambda() -> None:
    entries = (_entry(_ngram(0), count=3), _entry(_ngram(0, 1), count=1))
    config = _config(commonness_bias=1.0, lambda_curve=0.0, lambda_harmonic=0.0, lambda_accent=0.0)
    rng = default_rng(0)

    draws = [
        select_figure(
            entries,
            anchor=0,
            target_slope=0,
            scale_type=ScaleType.MAJOR,
            chord_pitch_classes=_TONIC_PITCH_CLASSES,
            weight=1.0,
            config=config,
            rng=rng,
        )
        for _ in range(4000)
    ]
    common_fraction = sum(1 for entry in draws if entry.figure == _ngram(0)) / len(draws)

    assert 0.71 < common_fraction < 0.79  # true count ratio is 3 / 4


def test_harmonic_lambda_prefers_chord_tone_figures() -> None:
    chord_tone_figure = _ngram(0, 2, 4)  # C E G over the tonic
    passing_figure = _ngram(0, 1, 2)  # C D E — D is a non-chord tone
    entries = (_entry(chord_tone_figure, count=1), _entry(passing_figure, count=1))
    config = _config(commonness_bias=1.0, lambda_curve=0.0, lambda_harmonic=8.0, lambda_accent=0.0)
    rng = default_rng(0)

    draws = [
        select_figure(
            entries,
            anchor=0,
            target_slope=0,
            scale_type=ScaleType.MAJOR,
            chord_pitch_classes=_TONIC_PITCH_CLASSES,
            weight=1.0,
            config=config,
            rng=rng,
        )
        for _ in range(200)
    ]

    assert sum(1 for entry in draws if entry.figure == chord_tone_figure) > 180
