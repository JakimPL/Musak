import pytest
from numpy.random import default_rng

from musak_model.synthetic.structure.form import (
    ClosingChoice,
    FormPrior,
    FormSampler,
    FormTree,
    PhraseNode,
    SegmentNode,
    VariationKind,
    WeightedSpan,
)
from musak_model.synthetic.structure.harmony_grammar import ClosingPattern
from musak_shared.elements import HarmonicFunction

_HALF = (HarmonicFunction.PREDOMINANT, HarmonicFunction.DOMINANT)
_AUTHENTIC = (HarmonicFunction.DOMINANT, HarmonicFunction.TONIC)


def _period_prior(*, repeat_probability: float = 0.0, variation_probability: float = 0.0) -> FormPrior:
    return FormPrior(
        phrase_lengths=(WeightedSpan(bars=4, weight=1.0),),
        segment_lengths=(WeightedSpan(bars=2, weight=1.0),),
        closings=(
            ClosingChoice(is_final=False, functions=_HALF, weight=1.0),
            ClosingChoice(is_final=True, functions=_AUTHENTIC, weight=1.0),
        ),
        repeat_probability=repeat_probability,
        variation_probability=variation_probability,
    )


def test_sampled_form_tiles_into_phrases_and_segments() -> None:
    form = FormSampler(_period_prior()).sample(bar_count=8, rng=default_rng(0))

    assert [(phrase.start_bar, phrase.bar_span) for phrase in form.phrases] == [(0, 4), (4, 4)]
    assert [(segment.start_bar, segment.bar_span) for segment in form.segments] == [(0, 2), (2, 2), (4, 2), (6, 2)]


def test_final_phrase_closes_authentic_others_open() -> None:
    form = FormSampler(_period_prior()).sample(bar_count=8, rng=default_rng(0))

    assert form.phrases[0].closing.terminal_function is HarmonicFunction.DOMINANT
    assert form.phrases[-1].closing.terminal_function is HarmonicFunction.TONIC


def test_sampler_is_deterministic_for_a_seed() -> None:
    sampler = FormSampler(_period_prior(repeat_probability=0.5, variation_probability=0.5))

    first = sampler.sample(bar_count=12, rng=default_rng(3))
    second = sampler.sample(bar_count=12, rng=default_rng(3))

    assert first == second


def test_no_repetition_makes_every_segment_a_fresh_class() -> None:
    form = FormSampler(_period_prior(repeat_probability=0.0)).sample(bar_count=8, rng=default_rng(1))

    assert all(segment.variation is VariationKind.FRESH for segment in form.segments)
    assert {segment.class_label for segment in form.segments} == {0, 1, 2, 3}


def test_full_repetition_reuses_earlier_classes() -> None:
    form = FormSampler(_period_prior(repeat_probability=1.0, variation_probability=0.5)).sample(
        bar_count=8, rng=default_rng(2)
    )

    # Only the first segment can be fresh; the rest restate earlier classes.
    assert form.segments[0].variation is VariationKind.FRESH
    assert all(segment.variation is not VariationKind.FRESH for segment in form.segments[1:])
    assert {segment.class_label for segment in form.segments} == {0}


def test_partition_absorbs_remainder_into_the_last_span() -> None:
    prior = FormPrior(
        phrase_lengths=(WeightedSpan(bars=4, weight=1.0),),
        segment_lengths=(WeightedSpan(bars=4, weight=1.0),),
        closings=(
            ClosingChoice(is_final=False, functions=_HALF, weight=1.0),
            ClosingChoice(is_final=True, functions=_AUTHENTIC, weight=1.0),
        ),
        repeat_probability=0.0,
        variation_probability=0.0,
    )

    form = FormSampler(prior).sample(bar_count=6, rng=default_rng(0))

    assert [(phrase.start_bar, phrase.bar_span) for phrase in form.phrases] == [(0, 4), (4, 2)]


def test_form_tree_rejects_non_tiling_segments() -> None:
    phrase = PhraseNode(start_bar=0, bar_span=4, closing=ClosingPattern(_AUTHENTIC))
    gap_segment = SegmentNode(start_bar=2, bar_span=2, class_label=0, variation=VariationKind.FRESH)

    with pytest.raises(ValueError, match="segments must tile"):
        FormTree(bar_count=4, segments=(gap_segment,), phrases=(phrase,))


def test_form_tree_rejects_phrase_misaligned_with_segments() -> None:
    segments = (
        SegmentNode(start_bar=0, bar_span=3, class_label=0, variation=VariationKind.FRESH),
        SegmentNode(start_bar=3, bar_span=1, class_label=1, variation=VariationKind.FRESH),
    )
    phrases = (
        PhraseNode(start_bar=0, bar_span=2, closing=ClosingPattern(_HALF)),
        PhraseNode(start_bar=2, bar_span=2, closing=ClosingPattern(_AUTHENTIC)),
    )

    with pytest.raises(ValueError, match="phrase boundaries must align"):
        FormTree(bar_count=4, segments=segments, phrases=phrases)
