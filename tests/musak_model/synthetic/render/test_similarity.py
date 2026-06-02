from fractions import Fraction

from numpy.random import default_rng

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabularyEntry, FigureVocabularyGroup
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.figure_selection import figure_log_scores, select_figure
from musak_model.synthetic.render.motif import MotifFigure, MotifSchema, ground_motif
from musak_model.synthetic.render.similarity import figure_edit_distance
from musak_model.tokens.schema import Hand, ScaleType

_CHORD_PITCH_CLASSES = frozenset({0, 4, 7})


def _ngram(*steps: int) -> FigureNGram:
    return FigureNGram(onsets=tuple((((step, 0),), Fraction(1)) for step in steps))


def _entry(figure: FigureNGram, count: int = 1) -> FigureVocabularyEntry:
    group = FigureVocabularyGroup(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, n=len(figure.onsets))
    return FigureVocabularyEntry(group=group, figure=figure, count=count)


def test_edit_distance_is_zero_for_identical_figures() -> None:
    assert figure_edit_distance(_ngram(0, 2), _ngram(0, 2)) == 0.0


def test_edit_distance_grades_a_single_step_then_caps_at_replacement_cost() -> None:
    assert figure_edit_distance(_ngram(0), _ngram(0)) == 0.0
    assert figure_edit_distance(_ngram(0), _ngram(1)) == 1.0
    # A distant single-note swap is bounded by delete+insert (the edit-distance flatness limit, coherence §14.6).
    assert figure_edit_distance(_ngram(0), _ngram(3)) == 2.0


def test_edit_distance_counts_insertion() -> None:
    assert figure_edit_distance(_ngram(0), _ngram(0, 2)) == 1.0


def test_similarity_pulls_selection_toward_the_intended_figure() -> None:
    entries = (_entry(_ngram(0)), _entry(_ngram(1)), _entry(_ngram(5)))
    config = RenderConfig.load().model_copy(update={"lambda_similarity": 5.0})

    selected = select_figure(
        entries,
        anchor=0,
        target_slope=0,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=_CHORD_PITCH_CLASSES,
        weight=1.0,
        config=config,
        rng=default_rng(0),
        intended=_ngram(5),
    )

    assert selected.figure == _ngram(5)


def test_zero_lambda_similarity_leaves_scores_unchanged() -> None:
    entries = (_entry(_ngram(0)), _entry(_ngram(5)))
    config = RenderConfig.load()

    without_intended = figure_log_scores(
        entries,
        anchor=0,
        target_slope=0,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=_CHORD_PITCH_CLASSES,
        weight=1.0,
        config=config,
    )
    with_intended = figure_log_scores(
        entries,
        anchor=0,
        target_slope=0,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=_CHORD_PITCH_CLASSES,
        weight=1.0,
        config=config,
        intended=_ngram(0),
    )

    assert list(without_intended) == list(with_intended)


def test_ground_motif_places_anchors_relative_to_base() -> None:
    schema = MotifSchema(
        (
            MotifFigure(slot_index=0, figure=_ngram(0), anchor_offset=0),
            MotifFigure(slot_index=2, figure=_ngram(0, 1), anchor_offset=2),
        )
    )

    grounded = ground_motif(schema, base_anchor=10)

    assert grounded[0].anchor == 10
    assert grounded[2].anchor == 12
    assert grounded[2].figure == _ngram(0, 1)
