from fractions import Fraction
from pathlib import Path

from numpy.random import default_rng

from musak_model.data.schema import Segment
from musak_model.generation.constraints import GenerationConstraints, GenerationConstraintState
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry, FigureVocabularyGroup
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.renderer import SurfaceRenderer, phrase_harmony
from musak_model.synthetic.structure.form import ClosingChoice, FormPrior, FormSampler, FormTree, WeightedSpan
from musak_model.synthetic.structure.harmony_grammar import HarmonyGrammarConfig, HarmonyGrammarSampler
from musak_model.synthetic.structure.meter import MetricalGrammarConfig, MetricalTreeSampler
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HoldToken, NoteToken, ScaleType
from musak_shared.elements import HarmonicFunction, degrees_for_function

_HALF = (HarmonicFunction.PREDOMINANT, HarmonicFunction.DOMINANT)
_AUTHENTIC = (HarmonicFunction.DOMINANT, HarmonicFunction.TONIC)
_SOURCE_FILE = Path("synthetic.mxl")
_TONIC_DEGREES = degrees_for_function(HarmonicFunction.TONIC, scale_size=7)
_DOMINANT_DEGREES = degrees_for_function(HarmonicFunction.DOMINANT, scale_size=7)


def _ngram(*steps: int) -> FigureNGram:
    return FigureNGram(onsets=tuple((((step, 0),), Fraction(1)) for step in steps))


def _figure_vocabulary() -> FigureVocabulary:
    entries: list[FigureVocabularyEntry] = []
    for hand in (Hand.RIGHT, Hand.LEFT):
        for figure, count in ((_ngram(0), 5), (_ngram(0, 1), 3), (_ngram(0, 2), 2), (_ngram(0, 2, 4), 2)):
            group = FigureVocabularyGroup(scale_type=ScaleType.MAJOR, hand=hand, n=len(figure.onsets))
            entries.append(FigureVocabularyEntry(group=group, figure=figure, count=count))
    return FigureVocabulary(entries=tuple(entries))


def _period_prior() -> FormPrior:
    return FormPrior(
        phrase_lengths=(WeightedSpan(bars=4, weight=1.0),),
        segment_lengths=(WeightedSpan(bars=2, weight=1.0),),
        closings=(
            ClosingChoice(is_final=False, functions=_HALF, weight=1.0),
            ClosingChoice(is_final=True, functions=_AUTHENTIC, weight=1.0),
        ),
        repeat_probability=0.0,
        variation_probability=0.0,
    )


def _harmony_sampler() -> HarmonyGrammarSampler:
    return HarmonyGrammarSampler(config=HarmonyGrammarConfig.load(), vocabulary=ChordVocabularyConfig.load())


def _renderer(
    duration_vocabulary: DurationVocabulary,
    *,
    metrical_overrides: dict[str, object] | None = None,
) -> SurfaceRenderer:
    metrical_config = MetricalGrammarConfig.load()
    if metrical_overrides is not None:
        metrical_config = metrical_config.model_copy(update=metrical_overrides)
    return SurfaceRenderer(
        config=RenderConfig.load(),
        metrical_sampler=MetricalTreeSampler(config=metrical_config),
        harmony_sampler=_harmony_sampler(),
        register_curve_sampler=RegisterCurveSampler(config=RegisterCurveConfig.load()),
        figure_vocabulary=_figure_vocabulary(),
        duration_vocabulary=duration_vocabulary,
        chord_vocabulary=ChordVocabularyConfig.load(),
    )


def _form(bar_count: int, seed: int) -> FormTree:
    return FormSampler(_period_prior()).sample(bar_count=bar_count, rng=default_rng(seed))


def _constraints(bar_count: int) -> GenerationConstraints:
    return GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=bar_count)


def _render(renderer: SurfaceRenderer, *, bar_count: int, seed: int) -> Segment:
    return renderer.render(
        time_numerator=4,
        time_denominator=4,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        form=_form(bar_count, seed),
        harmonic_slot_duration=Fraction(1),
        constraints=_constraints(bar_count),
        source_file=_SOURCE_FILE,
        rng=default_rng(seed),
    )


def _revalidate(segment: Segment, duration_vocabulary: DurationVocabulary, *, bar_count: int) -> None:
    state = GenerationConstraintState(constraints=_constraints(bar_count))
    for token in segment.tokens:
        state = state.apply(token, duration_vocabulary=duration_vocabulary)


def test_render_produces_a_constraint_valid_segment(duration_vocabulary: DurationVocabulary) -> None:
    segment = _render(_renderer(duration_vocabulary), bar_count=8, seed=0)

    assert segment.metadata.bar_count == 8
    assert any(isinstance(token, NoteToken) for token in segment.tokens)
    _revalidate(segment, duration_vocabulary, bar_count=8)


def test_render_is_deterministic_for_a_seed(duration_vocabulary: DurationVocabulary) -> None:
    renderer = _renderer(duration_vocabulary)

    first = _render(renderer, bar_count=8, seed=7)
    second = _render(renderer, bar_count=8, seed=7)

    assert first.tokens == second.tokens


def test_coarse_slots_yield_long_notes(duration_vocabulary: DurationVocabulary) -> None:
    renderer = _renderer(
        duration_vocabulary, metrical_overrides={"subdivision_probability": 1.0, "subdivision_decay": 0.0}
    )
    segment = _render(renderer, bar_count=4, seed=0)

    half_note_id = duration_vocabulary.require_duration_id(Fraction(1, 2))
    assert any(isinstance(token, NoteToken) and token.duration_id == half_note_id for token in segment.tokens)
    _revalidate(segment, duration_vocabulary, bar_count=4)


def test_tie_slots_become_held_notes(duration_vocabulary: DurationVocabulary) -> None:
    renderer = _renderer(
        duration_vocabulary,
        metrical_overrides={
            "subdivision_probability": 1.0,
            "subdivision_decay": 0.0,
            "rest_probability": 0.0,
            "tie_probability": 1.0,
        },
    )
    segment = _render(renderer, bar_count=4, seed=0)

    assert any(isinstance(token, HoldToken) for token in segment.tokens)
    _revalidate(segment, duration_vocabulary, bar_count=4)


def test_phrase_harmony_reproduces_the_period() -> None:
    tree = MetricalTreeSampler(
        config=MetricalGrammarConfig.load().model_copy(
            update={"subdivision_probability": 1.0, "subdivision_decay": 1.0}
        )
    ).sample(time_numerator=4, time_denominator=4, bar_count=8, rng=default_rng(0))
    frontier = tree.harmonic_frontier(Fraction(1))

    chords = phrase_harmony(
        frontier,
        _form(8, 0),
        harmony_sampler=_harmony_sampler(),
        scale_type=ScaleType.MAJOR,
        bar_duration=Fraction(1),
        rng=default_rng(0),
    )

    assert len(chords) == 8
    assert chords[3].root_degree in _DOMINANT_DEGREES  # antecedent half cadence
    assert chords[7].root_degree in _TONIC_DEGREES  # consequent authentic cadence
