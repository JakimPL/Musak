from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from numpy.random import default_rng

from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.generation.constraints import GenerationConstraints
from musak_model.harmony.decoding import ChordDecoderConfig
from musak_model.harmony.diatonic import diatonic_triads
from musak_model.harmony.schema import Chord
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.fitting.chord import ChordFitConfig
from musak_model.synthetic.processes.chord_track import (
    ChordTrackSampler,
    functional_transition_model,
)
from musak_model.tokens.schema import ScaleType


class HarmonicPlanProvider(Protocol):
    def plan_windows(
        self,
        *,
        constraints: GenerationConstraints,
        seed: int,
    ) -> tuple[HarmonicPlanWindow, ...]: ...


@dataclass(frozen=True)
class FunctionalHarmonicPlanProvider:
    chord_vocabulary: ChordVocabularyConfig
    chord_fit: ChordFitConfig
    chord_decoding: ChordDecoderConfig

    @classmethod
    def load(cls) -> FunctionalHarmonicPlanProvider:
        return cls(
            chord_vocabulary=ChordVocabularyConfig.load(),
            chord_fit=ChordFitConfig.load(),
            chord_decoding=ChordDecoderConfig.load(),
        )

    def plan_windows(
        self,
        *,
        constraints: GenerationConstraints,
        seed: int,
    ) -> tuple[HarmonicPlanWindow, ...]:
        if constraints.scale_type is None:
            raise ValueError("harmonic plan generation requires scale_type constraints")

        bounds = harmonic_plan_window_bounds(
            constraints=constraints,
            resolution=self.chord_decoding.resolution,
        )
        chords = _enabled_diatonic_triads(
            self.chord_vocabulary,
            scale_type=constraints.scale_type,
        )
        transition_model = functional_transition_model(
            chords,
            scale_type=constraints.scale_type,
            strength=self.chord_fit.functional_strength,
            self_transition_bias=self.chord_fit.self_transition_bias,
        )
        chord_track = ChordTrackSampler(transition_model).sample(
            length=len(bounds),
            rng=default_rng(seed),
        )
        return tuple(
            HarmonicPlanWindow(start=start, end=end, chord=chord)
            for (start, end), chord in zip(bounds, chord_track, strict=True)
        )


def harmonic_plan_window_bounds(
    *,
    constraints: GenerationConstraints,
    resolution: int,
) -> tuple[tuple[Fraction, Fraction], ...]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    window_duration = Fraction(1, resolution)
    windows: list[tuple[Fraction, Fraction]] = []
    for bar_index in range(constraints.bar_count):
        bar_start = constraints.bar_start(bar_index)
        bar_end = constraints.bar_end(bar_index)
        window_start = bar_start
        while window_start < bar_end:
            window_end = min(window_start + window_duration, bar_end)
            if window_end <= window_start:
                raise ValueError("harmonic plan window bounds must make forward progress")

            windows.append((window_start, window_end))
            window_start = window_end

    return tuple(windows)


def _enabled_diatonic_triads(
    chord_vocabulary: ChordVocabularyConfig,
    *,
    scale_type: ScaleType,
) -> tuple[Chord, ...]:
    enabled_qualities = set(chord_vocabulary.enabled_qualities())
    enabled_extensions = set(chord_vocabulary.enabled_extensions())
    return tuple(
        chord
        for chord in diatonic_triads(scale_type)
        if chord.quality in enabled_qualities and chord.extension in enabled_extensions
    )
