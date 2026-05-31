from musak_model.harmony.decoding.candidates import spellable_candidates
from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.base_durations import BaseDurationDistribution
from musak_model.synthetic.figures import AnchoredFigureVocabulary, FigureVocabulary
from musak_model.synthetic.fitting.artifacts import FittedChordTransitions, FittedGeneratorConfig
from musak_model.synthetic.processes.chord_track import (
    ChordTransitionModel,
    functional_transition_model,
    uniform_transition_model,
)
from musak_model.synthetic.substitution import FigureByChordModel
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType
from notebooks.utils.synthetic import SyntheticGenerationRequest, SyntheticInputs, _chord_transition_model


def _major_chords() -> tuple[Chord, ...]:
    return tuple(
        candidate.chord for candidate in spellable_candidates(ChordVocabularyConfig.load(), scale_type=ScaleType.MAJOR)
    )


def _empirical_major_model() -> ChordTransitionModel:
    tonic = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)
    dominant = Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR)
    return ChordTransitionModel(
        initial_distribution={tonic: 0.9, dominant: 0.1},
        transitions={tonic: {tonic: 0.1, dominant: 0.9}, dominant: {tonic: 0.95, dominant: 0.05}},
    )


def _inputs(fitted: FittedGeneratorConfig) -> SyntheticInputs:
    return SyntheticInputs(
        figure_vocabulary=FigureVocabulary(entries=()),
        anchored_figure_vocabulary=AnchoredFigureVocabulary(entries=()),
        base_duration_distribution=BaseDurationDistribution(weights_by_group={}),
        duration_vocabulary=DurationVocabulary(TokenizationConfig.load()),
        fitted=fitted,
        figure_by_chord_model=FigureByChordModel(),
    )


def _request(chord_model: str) -> SyntheticGenerationRequest:
    return SyntheticGenerationRequest(
        scale_root=0,
        scale_type="major",
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=4,
        chord_resolution=1,
        bar_count=4,
        seed=0,
        min_n=2,
        max_n=2,
        monophonic=True,
        lambda_curve=0.0,
        lambda_harm=0.0,
        lambda_accent=0.0,
        lambda_chord_figure=0.0,
        commonness_bias=1.0,
        max_resample_retries=4,
        arch_basis_count=3,
        arch_amplitude=1.0,
        arch_decay=1.0,
        ou_theta=0.2,
        ou_sigma=1.0,
        baseline_logit=0.0,
        metric_gain=1.0,
        metric_exponent=1.0,
        envelope_basis_count=3,
        envelope_amplitude=0.5,
        envelope_decay=1.0,
        co_activity_strength=0.7,
        activity_right=0.9,
        activity_left=0.9,
        sync_strength=0.0,
        self_transition_bias=0.25,
        functional_strength=0.7,
        chord_model=chord_model,
        use_constraints=False,
        minimum_duration="None",
        allow_dotted=True,
        max_notes_per_hand=None,
        max_onset_span=None,
        max_gap=None,
        max_span=None,
    )


def test_empirical_chord_model_uses_the_baked_transition_model() -> None:
    model = _empirical_major_model()
    inputs = _inputs(
        FittedGeneratorConfig(chord_transitions={ScaleType.MAJOR: FittedChordTransitions.from_model(model)})
    )

    selected = _chord_transition_model(
        _request("empirical"), inputs, chords=_major_chords(), scale_type=ScaleType.MAJOR
    )

    assert selected == model


def test_empirical_chord_model_falls_back_to_functional_when_unfitted() -> None:
    chords = _major_chords()
    inputs = _inputs(FittedGeneratorConfig())

    selected = _chord_transition_model(_request("empirical"), inputs, chords=chords, scale_type=ScaleType.MAJOR)

    assert selected == functional_transition_model(
        chords, scale_type=ScaleType.MAJOR, strength=0.7, self_transition_bias=0.25
    )


def test_uniform_chord_model_uses_uniform_transitions() -> None:
    chords = _major_chords()

    selected = _chord_transition_model(
        _request("uniform"), _inputs(FittedGeneratorConfig()), chords=chords, scale_type=ScaleType.MAJOR
    )

    assert selected == uniform_transition_model(chords, self_transition_bias=0.25)
