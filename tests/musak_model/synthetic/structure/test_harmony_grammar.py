from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from numpy.random import default_rng
from pydantic import ValidationError

from musak_model.harmony.diatonic import natural_triad
from musak_model.harmony.schema import DEFAULT_CHORD_EXTENSION, ChordExtension
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.structure.harmony_grammar import (
    ClosingPattern,
    HarmonyGrammarConfig,
    HarmonyGrammarSampler,
    HarmonyNode,
)
from musak_model.tokens.schema import ScaleType
from musak_shared.elements import HarmonicFunction, degrees_for_function

_TONIC_DEGREES = degrees_for_function(HarmonicFunction.TONIC, scale_size=7)
_PREDOMINANT_DEGREES = degrees_for_function(HarmonicFunction.PREDOMINANT, scale_size=7)
_DOMINANT_DEGREES = degrees_for_function(HarmonicFunction.DOMINANT, scale_size=7)
_TONIC_SUBSTITUTE_DEGREES = tuple(degree for degree in _TONIC_DEGREES if degree != 1)

_AUTHENTIC = ClosingPattern((HarmonicFunction.DOMINANT, HarmonicFunction.TONIC))
_PLAGAL = ClosingPattern((HarmonicFunction.PREDOMINANT, HarmonicFunction.TONIC))
_HALF = ClosingPattern((HarmonicFunction.PREDOMINANT, HarmonicFunction.DOMINANT))


def _default_sampler() -> HarmonyGrammarSampler:
    return HarmonyGrammarSampler(config=HarmonyGrammarConfig.load(), vocabulary=ChordVocabularyConfig.load())


def _vocabulary_with_seventh() -> ChordVocabularyConfig:
    vocabulary = ChordVocabularyConfig.load()
    seventh = vocabulary.extensions[ChordExtension.SEVENTH].model_copy(update={"enabled": True})
    return vocabulary.model_copy(update={"extensions": {**vocabulary.extensions, ChordExtension.SEVENTH: seventh}})


def _walk(node: HarmonyNode) -> Iterator[HarmonyNode]:
    yield node
    for child in node.children:
        yield from _walk(child)


@dataclass(frozen=True)
class _CountCase:
    slot_count: int
    closing: ClosingPattern


_COUNT_CASES = tuple(
    _CountCase(slot_count, closing) for slot_count in (1, 2, 3, 4, 8) for closing in (_AUTHENTIC, _PLAGAL, _HALF)
)


@pytest.mark.parametrize("case", _COUNT_CASES, ids=lambda case: f"{case.slot_count}-{case.closing.functions}")
def test_chords_count_matches_slot_count(case: _CountCase) -> None:
    tree = _default_sampler().sample(
        slot_count=case.slot_count, scale_type=ScaleType.MAJOR, closing=case.closing, rng=default_rng(0)
    )

    assert len(tree.chords()) == case.slot_count


def test_authentic_suffix_ends_dominant_then_tonic() -> None:
    sampler = _default_sampler()

    for seed in range(20):
        chords = sampler.sample(
            slot_count=4, scale_type=ScaleType.MAJOR, closing=_AUTHENTIC, rng=default_rng(seed)
        ).chords()

        assert chords[-2].root_degree in _DOMINANT_DEGREES
        assert chords[-1].root_degree in _TONIC_DEGREES


def test_plagal_suffix_ends_subdominant_then_tonic() -> None:
    sampler = _default_sampler()

    for seed in range(20):
        chords = sampler.sample(
            slot_count=4, scale_type=ScaleType.MAJOR, closing=_PLAGAL, rng=default_rng(seed)
        ).chords()

        assert chords[-2].root_degree in _PREDOMINANT_DEGREES
        assert chords[-1].root_degree in _TONIC_DEGREES


def test_half_suffix_ends_on_a_dominant() -> None:
    sampler = _default_sampler()

    for seed in range(20):
        chords = sampler.sample(slot_count=4, scale_type=ScaleType.MAJOR, closing=_HALF, rng=default_rng(seed)).chords()

        assert chords[-1].root_degree in _DOMINANT_DEGREES


def test_triads_only_when_the_vocabulary_disables_extensions() -> None:
    chords = (
        _default_sampler()
        .sample(slot_count=8, scale_type=ScaleType.MAJOR, closing=_AUTHENTIC, rng=default_rng(0))
        .chords()
    )

    assert all(chord.extension is DEFAULT_CHORD_EXTENSION for chord in chords)


def test_seventh_chords_appear_only_when_enabled_in_the_vocabulary() -> None:
    sampler = HarmonyGrammarSampler(
        config=HarmonyGrammarConfig.load().model_copy(update={"extension_decay": 1.0}),
        vocabulary=_vocabulary_with_seventh(),
    )

    has_seventh = any(
        chord.extension is ChordExtension.SEVENTH
        for seed in range(20)
        for chord in sampler.sample(
            slot_count=6, scale_type=ScaleType.MAJOR, closing=_AUTHENTIC, rng=default_rng(seed)
        ).chords()
    )

    assert has_seventh


def test_tonic_function_realizes_diatonic_substitutes() -> None:
    sampler = _default_sampler()

    has_substitute = any(
        chord.root_degree in _TONIC_SUBSTITUTE_DEGREES
        for seed in range(20)
        for chord in sampler.sample(
            slot_count=6, scale_type=ScaleType.MAJOR, closing=_AUTHENTIC, rng=default_rng(seed)
        ).chords()
    )

    assert has_substitute


def test_leaves_carry_chords_and_internal_nodes_do_not() -> None:
    tree = _default_sampler().sample(slot_count=8, scale_type=ScaleType.MAJOR, closing=_AUTHENTIC, rng=default_rng(2))

    for node in _walk(tree):
        assert node.is_leaf == (node.chord is not None)


def test_sampler_is_deterministic_for_a_seed() -> None:
    sampler = _default_sampler()

    first = sampler.sample(slot_count=8, scale_type=ScaleType.MAJOR, closing=_HALF, rng=default_rng(9))
    second = sampler.sample(slot_count=8, scale_type=ScaleType.MAJOR, closing=_HALF, rng=default_rng(9))

    assert first == second


def test_closing_pattern_rejects_empty_suffix() -> None:
    with pytest.raises(ValueError, match="closing pattern"):
        ClosingPattern(())


def test_node_rejects_leaf_without_chord() -> None:
    with pytest.raises(ValueError, match="leaf"):
        HarmonyNode(function=HarmonicFunction.TONIC)


def test_node_rejects_internal_with_chord() -> None:
    tonic = natural_triad(ScaleType.MAJOR, 1)
    leaf = HarmonyNode(function=HarmonicFunction.TONIC, chord=tonic)

    with pytest.raises(ValueError, match="must not carry a chord"):
        HarmonyNode(function=HarmonicFunction.TONIC, children=(leaf,), chord=tonic)


def test_sample_rejects_non_positive_slot_count() -> None:
    with pytest.raises(ValueError, match="slot_count"):
        _default_sampler().sample(slot_count=0, scale_type=ScaleType.MAJOR, closing=_AUTHENTIC, rng=default_rng(0))


def test_config_rejects_out_of_range_value() -> None:
    with pytest.raises(ValidationError):
        HarmonyGrammarConfig(prepare_probability=1.5, extension_decay=0.4)
