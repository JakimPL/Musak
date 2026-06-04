from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from numpy.random import default_rng

from musak_model.conditioning.harmony.planner import (
    HarmonicPlan,
    HarmonicPlannerConfig,
    harmonic_plan_candidate_chords,
    plan_harmony,
)
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.generation.constraints import GenerationConstraints
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.fitting.chord import ChordFitConfig
from musak_model.synthetic.processes.chord_track import functional_transition_model


class HarmonicPlanProvider(Protocol):
    def plan(
        self,
        *,
        constraints: GenerationConstraints,
        seed: int,
    ) -> HarmonicPlan: ...

    def plan_windows(
        self,
        *,
        constraints: GenerationConstraints,
        seed: int,
    ) -> tuple[HarmonicPlanWindow, ...]: ...


@dataclass(frozen=True)
class FiniteHorizonHarmonicPlanProvider:
    chord_vocabulary: ChordVocabularyConfig
    chord_fit: ChordFitConfig
    planner: HarmonicPlannerConfig

    @classmethod
    def load(cls) -> FiniteHorizonHarmonicPlanProvider:
        return cls(
            chord_vocabulary=ChordVocabularyConfig.load(),
            chord_fit=ChordFitConfig.load(),
            planner=HarmonicPlannerConfig.load(),
        )

    def plan_windows(
        self,
        *,
        constraints: GenerationConstraints,
        seed: int,
    ) -> tuple[HarmonicPlanWindow, ...]:
        return self.plan(constraints=constraints, seed=seed).windows

    def plan(
        self,
        *,
        constraints: GenerationConstraints,
        seed: int,
    ) -> HarmonicPlan:
        if constraints.scale_type is None:
            raise ValueError("harmonic plan generation requires scale_type constraints")

        candidates = harmonic_plan_candidate_chords(
            self.chord_vocabulary,
            scale_type=constraints.scale_type,
            candidate_limit=self.planner.candidate_limit_per_slot,
        )
        transition_model = functional_transition_model(
            candidates,
            scale_type=constraints.scale_type,
            strength=self.chord_fit.functional_strength,
            self_transition_bias=self.chord_fit.self_transition_bias,
        )
        return plan_harmony(
            constraints=constraints,
            scale_type=constraints.scale_type,
            chord_vocabulary=self.chord_vocabulary,
            transition_model=transition_model,
            config=self.planner,
            rng=default_rng(seed),
        )
