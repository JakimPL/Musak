from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from musak_model.harmony.diatonic import natural_triad
from musak_model.harmony.schema import DEFAULT_CHORD_EXTENSION, Chord
from musak_model.harmony.vocabulary import TRIAD_INTERVAL_COUNT, ChordVocabularyConfig
from musak_model.paths import HARMONY_GRAMMAR_CONFIG_PATH
from musak_model.tokens.schema import SCALE_INTERVALS, ScaleType
from musak_shared.elements import HarmonicFunction, degrees_for_function
from musak_shared.files import load_yaml_config

_PREPARER: Final[dict[HarmonicFunction, HarmonicFunction]] = {
    HarmonicFunction.TONIC: HarmonicFunction.DOMINANT,
    HarmonicFunction.DOMINANT: HarmonicFunction.PREDOMINANT,
}


@dataclass(frozen=True)
class ClosingPattern:
    functions: tuple[HarmonicFunction, ...]

    def __post_init__(self) -> None:
        if not self.functions:
            raise ValueError("a closing pattern must have at least one function")

    @property
    def terminal_function(self) -> HarmonicFunction:
        return self.functions[-1]


@dataclass(frozen=True)
class HarmonyNode:
    function: HarmonicFunction
    children: tuple[HarmonyNode, ...] = ()
    chord: Chord | None = None

    def __post_init__(self) -> None:
        if self.children and self.chord is not None:
            raise ValueError("internal nodes must not carry a chord")

        if not self.children and self.chord is None:
            raise ValueError("a leaf must carry a chord")

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def chords(self) -> tuple[Chord, ...]:
        if self.chord is not None:
            return (self.chord,)
        return tuple(chord for child in self.children for chord in child.chords())


class HarmonyGrammarConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prepare_probability: float = Field(ge=0.0, le=1.0)
    extension_decay: float = Field(ge=0.0, le=1.0)

    @classmethod
    def load(cls, path: Path = HARMONY_GRAMMAR_CONFIG_PATH) -> HarmonyGrammarConfig:
        return cls.model_validate(load_yaml_config(path))


@dataclass(frozen=True)
class HarmonyGrammarSampler:
    config: HarmonyGrammarConfig
    vocabulary: ChordVocabularyConfig

    def sample(
        self,
        *,
        slot_count: int,
        scale_type: ScaleType,
        closing: ClosingPattern,
        rng: Generator,
    ) -> HarmonyNode:
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")

        suffix = closing.functions[-slot_count:]
        suffix_nodes = tuple(
            HarmonyNode(function=function, chord=self._sample_chord(function, scale_type, rng)) for function in suffix
        )
        body_slots = slot_count - len(suffix_nodes)
        if body_slots <= 0:
            if len(suffix_nodes) == 1:
                return suffix_nodes[0]
            return HarmonyNode(function=closing.terminal_function, children=suffix_nodes)

        body = self._derive_region(HarmonicFunction.TONIC, body_slots, scale_type, rng)
        return HarmonyNode(function=closing.terminal_function, children=(body, *suffix_nodes))

    def _derive_region(
        self,
        function: HarmonicFunction,
        span: int,
        scale_type: ScaleType,
        rng: Generator,
    ) -> HarmonyNode:
        if span == 1:
            return HarmonyNode(function=function, chord=self._sample_chord(function, scale_type, rng))

        preparer = _PREPARER.get(function)
        if preparer is not None and rng.random() < self.config.prepare_probability:
            left_function = preparer
        else:
            left_function = function

        split = int(rng.integers(1, span))
        left = self._derive_region(left_function, split, scale_type, rng)
        right = self._derive_region(function, span - split, scale_type, rng)
        return HarmonyNode(function=function, children=(left, right))

    def _sample_chord(self, function: HarmonicFunction, scale_type: ScaleType, rng: Generator) -> Chord:
        chords, probabilities = self._candidates(function, scale_type)
        return chords[int(rng.choice(len(chords), p=probabilities))]

    def _candidates(
        self, function: HarmonicFunction, scale_type: ScaleType
    ) -> tuple[tuple[Chord, ...], NDArray[np.float64]]:
        scale_size = len(SCALE_INTERVALS[scale_type])
        chords: list[Chord] = []
        weights: list[float] = []
        for degree in degrees_for_function(function, scale_size=scale_size):
            triad = natural_triad(scale_type, degree)
            for extension in self.vocabulary.enabled_extensions():
                if extension is DEFAULT_CHORD_EXTENSION:
                    chord = triad
                else:
                    chord = triad.model_copy(update={"extension": extension})
                members = self.vocabulary.extension_definition(extension).members
                chords.append(chord)
                weights.append(self.config.extension_decay ** (members - TRIAD_INTERVAL_COUNT))

        probabilities = np.array(weights, dtype=np.float64)
        return tuple(chords), probabilities / probabilities.sum()
