from pathlib import Path

from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.synthetic.fitting.artifacts import FittedChordTransitions, FittedGeneratorConfig
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldOverride
from musak_model.synthetic.processes.chord_track import ChordTransitionModel
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveOverride
from musak_model.tokens.schema import Hand, ScaleType


def _register_override() -> RegisterCurveOverride:
    return RegisterCurveOverride(
        scale_type=ScaleType.MAJOR,
        hand=Hand.RIGHT,
        config=RegisterCurveConfig(arch_basis_count=3, arch_amplitude=2.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=0.5),
    )


def _accent_override() -> AccentFieldOverride:
    return AccentFieldOverride(
        scale_type=ScaleType.MAJOR,
        hand=Hand.LEFT,
        config=AccentFieldConfig(
            baseline_logit=-0.5,
            metric_gain=2.0,
            metric_exponent=1.0,
            envelope_basis_count=3,
            envelope_amplitude=0.5,
            envelope_decay=1.0,
        ),
    )


def test_fitted_generator_config_round_trips_through_json(tmp_path: Path) -> None:
    config = FittedGeneratorConfig(
        register_overrides=(_register_override(),),
        accent_overrides=(_accent_override(),),
        chord_transitions={ScaleType.MAJOR: FittedChordTransitions.from_model(_transition_model())},
    )
    path = tmp_path / "fitted" / "generator.json"

    config.write(path)

    assert FittedGeneratorConfig.read(path) == config


def test_fitted_transition_model_round_trips_to_runtime_object(tmp_path: Path) -> None:
    transition_model = _transition_model()
    config = FittedGeneratorConfig(
        chord_transitions={ScaleType.MAJOR: FittedChordTransitions.from_model(transition_model)},
    )
    path = tmp_path / "fitted" / "generator.json"

    config.write(path)
    restored = FittedGeneratorConfig.read(path)

    assert restored.chord_transition_model(ScaleType.MAJOR) == transition_model
    assert restored.chord_transition_model(ScaleType.HARMONIC_MINOR) is None


def test_fitted_generator_config_defaults_to_no_overrides() -> None:
    config = FittedGeneratorConfig()

    assert config.register_overrides == ()
    assert config.accent_overrides == ()
    assert config.chord_transitions == {}
    assert config.chord_transition_model(ScaleType.MAJOR) is None


def _transition_model() -> ChordTransitionModel:
    tonic = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)
    dominant = Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR)
    return ChordTransitionModel(
        initial_distribution={tonic: 0.6, dominant: 0.4},
        transitions={tonic: {tonic: 0.3, dominant: 0.7}, dominant: {tonic: 0.8, dominant: 0.2}},
    )
