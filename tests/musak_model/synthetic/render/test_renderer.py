from fractions import Fraction
from pathlib import Path

from numpy.random import default_rng

from musak_model.data.schema import Segment
from musak_model.generation.constraints import GenerationConstraints, GenerationConstraintState
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.base_durations import BaseDurationDistribution
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry, FigureVocabularyGroup
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler
from musak_model.synthetic.processes.density import RhythmicDensityConfig, RhythmicDensitySampler
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.motif import MotifConfig
from musak_model.synthetic.render.renderer import SurfaceRenderer, class_metrical_tree, phrase_harmony
from musak_model.synthetic.structure.form import ClosingChoice, FormPrior, FormSampler, FormTree, WeightedSpan
from musak_model.synthetic.structure.harmony_grammar import HarmonyGrammarConfig, HarmonyGrammarSampler
from musak_model.synthetic.structure.meter import (
    MetricalGrammarConfig,
    MetricalLeafType,
    MetricalNode,
    MetricalTreeSampler,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, NoteToken, RestToken, ScaleType, Token
from musak_shared.elements import HarmonicFunction, degrees_for_function

_HALF = (HarmonicFunction.PREDOMINANT, HarmonicFunction.DOMINANT)
_AUTHENTIC = (HarmonicFunction.DOMINANT, HarmonicFunction.TONIC)
_SOURCE_FILE = Path("synthetic.mxl")
_TONIC_DEGREES = degrees_for_function(HarmonicFunction.TONIC, scale_size=7)
_DOMINANT_DEGREES = degrees_for_function(HarmonicFunction.DOMINANT, scale_size=7)
_BASE_DURATIONS = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))


def _ngram(*steps: int) -> FigureNGram:
    return FigureNGram(onsets=tuple((((step, 0),), Fraction(1)) for step in steps))


def _figure_vocabulary() -> FigureVocabulary:
    entries: list[FigureVocabularyEntry] = []
    for hand in (Hand.RIGHT, Hand.LEFT):
        for figure, count in ((_ngram(0), 5), (_ngram(0, 1), 3), (_ngram(0, 2), 2), (_ngram(0, 2, 4), 2)):
            group = FigureVocabularyGroup(scale_type=ScaleType.MAJOR, hand=hand, n=len(figure.onsets))
            entries.append(FigureVocabularyEntry(group=group, figure=figure, count=count))
    return FigureVocabulary(entries=tuple(entries))


def _base_durations() -> BaseDurationDistribution:
    weights = {}
    for hand in (Hand.RIGHT, Hand.LEFT):
        weights[(ScaleType.MAJOR, hand, 1)] = ((Fraction(1, 4), 1), (Fraction(1, 2), 1))
        weights[(ScaleType.MAJOR, hand, 2)] = ((Fraction(1, 8), 1), (Fraction(1, 4), 1))
        weights[(ScaleType.MAJOR, hand, 3)] = ((Fraction(1, 8), 1),)
    return BaseDurationDistribution(weights_by_group=weights)


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


def _accent_config(*, baseline_logit: float) -> AccentFieldConfig:
    return AccentFieldConfig(
        baseline_logit=baseline_logit,
        metric_gain=0.0,
        metric_exponent=1.0,
        envelope_basis_count=1,
        envelope_amplitude=0.0,
        envelope_decay=1.0,
    )


def _renderer(
    duration_vocabulary: DurationVocabulary,
    *,
    accent_config: AccentFieldConfig | None = None,
    lambda_similarity: float = 0.0,
    variation_budget: float = 0.0,
    maximum_transpose: int = 0,
    melodic_continuity: float | None = None,
) -> SurfaceRenderer:
    config_update: dict[str, float] = {"lambda_similarity": lambda_similarity}
    if melodic_continuity is not None:
        config_update["melodic_continuity"] = melodic_continuity

    return SurfaceRenderer(
        config=RenderConfig.load().model_copy(update=config_update),
        metrical_sampler=MetricalTreeSampler(config=MetricalGrammarConfig.load()),
        harmony_sampler=_harmony_sampler(),
        register_curve_sampler=RegisterCurveSampler(config=RegisterCurveConfig.load()),
        figure_vocabulary=_figure_vocabulary(),
        duration_vocabulary=duration_vocabulary,
        chord_vocabulary=ChordVocabularyConfig.load(),
        base_duration_distribution=_base_durations(),
        rhythmic_density_sampler=RhythmicDensitySampler(
            config=RhythmicDensityConfig(amplitude=1.0, basis_count=2, decay=1.0)
        ),
        accent_field_sampler=AccentFieldSampler(config=accent_config or AccentFieldConfig.load()),
        grid_denominator=4,
        motif_config=MotifConfig.load().model_copy(
            update={"variation_budget": variation_budget, "maximum_transpose": maximum_transpose}
        ),
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


def _repeat_form(*, bar_count: int, seed: int) -> FormTree:
    prior = FormPrior(
        phrase_lengths=(WeightedSpan(bars=4, weight=1.0),),
        segment_lengths=(WeightedSpan(bars=2, weight=1.0),),
        closings=(
            ClosingChoice(is_final=False, functions=_HALF, weight=1.0),
            ClosingChoice(is_final=True, functions=_AUTHENTIC, weight=1.0),
        ),
        repeat_probability=1.0,
        variation_probability=0.0,
    )
    return FormSampler(prior).sample(bar_count=bar_count, rng=default_rng(seed))


def _render_form(renderer: SurfaceRenderer, form: FormTree, *, bar_count: int, seed: int) -> Segment:
    return renderer.render(
        time_numerator=4,
        time_denominator=4,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        form=form,
        harmonic_slot_duration=Fraction(1),
        constraints=_constraints(bar_count),
        source_file=_SOURCE_FILE,
        rng=default_rng(seed),
    )


def _bar_token_groups(segment: Segment) -> list[list[Token]]:
    groups: list[list[Token]] = []
    current: list[Token] = []
    for token in segment.tokens:
        if isinstance(token, BarToken):
            groups.append(current)
            current = []
        elif not isinstance(token, EndToken):
            current.append(token)

    return groups


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


def test_note_durations_stay_on_corpus_base_durations(duration_vocabulary: DurationVocabulary) -> None:
    segment = _render(_renderer(duration_vocabulary), bar_count=8, seed=0)

    note_durations = {
        duration_vocabulary.id_to_fraction(token.duration_id)
        for token in segment.tokens
        if isinstance(token, NoteToken)
    }

    assert note_durations
    assert note_durations <= set(_BASE_DURATIONS)


def _rest_duration_fraction(segment: Segment, duration_vocabulary: DurationVocabulary) -> float:
    rest = sum(
        (
            duration_vocabulary.id_to_fraction(token.duration_id)
            for token in segment.tokens
            if isinstance(token, RestToken)
        ),
        Fraction(0),
    )
    note = sum(
        (
            duration_vocabulary.id_to_fraction(token.duration_id)
            for token in segment.tokens
            if isinstance(token, NoteToken)
        ),
        Fraction(0),
    )
    total = rest + note
    return float(rest / total) if total > 0 else 0.0


def test_render_emits_rests_with_default_config(duration_vocabulary: DurationVocabulary) -> None:
    segment = _render(_renderer(duration_vocabulary), bar_count=8, seed=0)

    assert any(isinstance(token, RestToken) for token in segment.tokens)


def test_lower_baseline_logit_increases_rests(duration_vocabulary: DurationVocabulary) -> None:
    sparse = _render(
        _renderer(duration_vocabulary, accent_config=_accent_config(baseline_logit=-6.0)), bar_count=8, seed=0
    )
    dense = _render(
        _renderer(duration_vocabulary, accent_config=_accent_config(baseline_logit=6.0)), bar_count=8, seed=0
    )

    assert _rest_duration_fraction(sparse, duration_vocabulary) > _rest_duration_fraction(dense, duration_vocabulary)
    _revalidate(sparse, duration_vocabulary, bar_count=8)
    _revalidate(dense, duration_vocabulary, bar_count=8)


def test_repeats_reuse_motif_at_high_similarity(duration_vocabulary: DurationVocabulary) -> None:
    renderer = _renderer(duration_vocabulary, lambda_similarity=50.0)
    form = _repeat_form(bar_count=4, seed=0)
    assert [segment.class_label for segment in form.segments] == [0, 0]

    segment = _render_form(renderer, form, bar_count=4, seed=0)
    bars = _bar_token_groups(segment)

    assert len(bars) == 4
    assert bars[0:2] == bars[2:4]


def test_similarity_changes_output_for_repeats(duration_vocabulary: DurationVocabulary) -> None:
    form = _repeat_form(bar_count=4, seed=0)

    inert = _render_form(_renderer(duration_vocabulary, lambda_similarity=0.0), form, bar_count=4, seed=0)
    engaged = _render_form(_renderer(duration_vocabulary, lambda_similarity=50.0), form, bar_count=4, seed=0)

    assert inert.tokens != engaged.tokens


def test_continuity_anchor_blends_register_and_carried(duration_vocabulary: DurationVocabulary) -> None:
    half = _renderer(duration_vocabulary, melodic_continuity=0.5)
    assert half._continuity_anchor(10, None) == 10  # no carried pitch yet → register anchor
    assert half._continuity_anchor(10, 20) == 15  # halfway between register and carried
    full = _renderer(duration_vocabulary, melodic_continuity=1.0)
    assert full._continuity_anchor(10, 20) == 20  # fully connected → previous ending pitch


def test_melodic_continuity_changes_output(duration_vocabulary: DurationVocabulary) -> None:
    connected = _render(_renderer(duration_vocabulary, melodic_continuity=1.0), bar_count=8, seed=0)
    reset = _render(_renderer(duration_vocabulary, melodic_continuity=0.0), bar_count=8, seed=0)

    assert connected.tokens != reset.tokens


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


def test_restated_segments_share_rhythm() -> None:
    prior = FormPrior(
        phrase_lengths=(WeightedSpan(bars=4, weight=1.0),),
        segment_lengths=(WeightedSpan(bars=2, weight=1.0),),
        closings=(
            ClosingChoice(is_final=False, functions=_HALF, weight=1.0),
            ClosingChoice(is_final=True, functions=_AUTHENTIC, weight=1.0),
        ),
        repeat_probability=1.0,
        variation_probability=0.0,
    )
    form = FormSampler(prior).sample(bar_count=4, rng=default_rng(0))
    assert [segment.class_label for segment in form.segments] == [0, 0]

    sampler = MetricalTreeSampler(config=MetricalGrammarConfig.load())
    tree = class_metrical_tree(sampler, form, time_numerator=4, time_denominator=4, rng=default_rng(3))

    def leaf_signature(bar: MetricalNode) -> list[tuple[Fraction, float, MetricalLeafType | None]]:
        return [(leaf.duration, leaf.weight, leaf.leaf_type) for leaf in bar.leaves()]

    assert [leaf_signature(bar) for bar in tree.bars[0:2]] == [leaf_signature(bar) for bar in tree.bars[2:4]]
