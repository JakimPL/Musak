from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from musak_model.n_grams.profile.artifacts import FIGURE_ALL_DIR_NAME, FIGURE_DIR_NAME
from musak_model.n_grams.profile.chord.schema import chord_from_key, chord_to_key
from musak_model.synthetic.processes.accent import AccentFieldOverride
from musak_model.synthetic.processes.chord_track import ChordTransitionModel
from musak_model.synthetic.processes.pitch import RegisterCurveOverride
from musak_model.synthetic.structure.form import FormPrior
from musak_model.tokens.schema import ScaleType

FITTED_GENERATOR_CONFIG_NAME: Final[str] = "fitted_generator.json"


class FittedChordTransitions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    initial: dict[str, float]
    transitions: dict[str, dict[str, float]]

    @classmethod
    def from_model(cls, model: ChordTransitionModel) -> FittedChordTransitions:
        return cls(
            initial={chord_to_key(chord): probability for chord, probability in model.initial_distribution.items()},
            transitions={
                chord_to_key(source): {
                    chord_to_key(destination): probability for destination, probability in row.items()
                }
                for source, row in model.transitions.items()
            },
        )

    def to_model(self) -> ChordTransitionModel:
        return ChordTransitionModel(
            initial_distribution={chord_from_key(key): probability for key, probability in self.initial.items()},
            transitions={
                chord_from_key(source): {
                    chord_from_key(destination): probability for destination, probability in row.items()
                }
                for source, row in self.transitions.items()
            },
        )


class FittedGeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    register_overrides: tuple[RegisterCurveOverride, ...] = ()
    accent_overrides: tuple[AccentFieldOverride, ...] = ()
    chord_transitions: dict[ScaleType, FittedChordTransitions] = {}
    form_priors: dict[ScaleType, FormPrior] = {}

    def chord_transition_model(self, scale_type: ScaleType) -> ChordTransitionModel | None:
        fitted = self.chord_transitions.get(scale_type)
        return fitted.to_model() if fitted is not None else None

    def form_prior(self, scale_type: ScaleType) -> FormPrior | None:
        return self.form_priors.get(scale_type)

    @classmethod
    def read(cls, path: Path) -> FittedGeneratorConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def resolve_fitted_generator_config_path(path: Path) -> Path | None:
    candidates = (
        path / FITTED_GENERATOR_CONFIG_NAME,
        path / FIGURE_ALL_DIR_NAME / FITTED_GENERATOR_CONFIG_NAME,
        path / FIGURE_DIR_NAME / FIGURE_ALL_DIR_NAME / FITTED_GENERATOR_CONFIG_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None
