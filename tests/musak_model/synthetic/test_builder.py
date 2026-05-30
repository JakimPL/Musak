from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.base_durations import BaseDurationDistribution
from musak_model.synthetic.builder import build_segment_generator
from musak_model.synthetic.figures import FigureVocabulary
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.synthetic.processes.chord_track import uniform_transition_model
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig
from musak_model.synthetic.processes.pitch import RegisterCurveConfig
from musak_model.synthetic.substitution import SegmentGenerator, SubstitutionConfig
from musak_model.tokens.duration import DurationVocabulary


def test_build_segment_generator_wires_components(duration_vocabulary: DurationVocabulary) -> None:
    substitution_config = SubstitutionConfig(
        lambda_curve=0.0,
        lambda_harm=0.0,
        lambda_accent=0.0,
        lambda_chord_figure=0.0,
        commonness_bias=1.0,
        max_resample_retries=4,
        monophonic=True,
    )
    register_curve_config = RegisterCurveConfig(
        arch_basis_count=3, arch_amplitude=1.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=1.0
    )
    accent_field_config = AccentFieldConfig(
        baseline_logit=0.0,
        metric_gain=0.0,
        metric_exponent=1.0,
        envelope_basis_count=3,
        envelope_amplitude=0.0,
        envelope_decay=1.0,
    )
    hand_coupling_config = HandCouplingConfig(
        co_activity_strength=0.5, activity_right=0.9, activity_left=0.9, sync_strength=0.0
    )
    chord_vocabulary = ChordVocabularyConfig.load()
    chord_transition_model = uniform_transition_model(
        (Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),)
    )
    figure_vocabulary = FigureVocabulary(entries=())
    base_duration_distribution = BaseDurationDistribution(weights_by_group={})

    generator = build_segment_generator(
        substitution_config=substitution_config,
        register_curve_config=register_curve_config,
        accent_field_config=accent_field_config,
        hand_coupling_config=hand_coupling_config,
        chord_transition_model=chord_transition_model,
        chord_vocabulary=chord_vocabulary,
        figure_vocabulary=figure_vocabulary,
        base_duration_distribution=base_duration_distribution,
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2, 3),
    )

    assert isinstance(generator, SegmentGenerator)
    assert generator.substitution_config is substitution_config
    assert generator.register_curve_sampler.config is register_curve_config
    assert generator.accent_field_sampler.config is accent_field_config
    assert generator.hand_coupling_sampler.config is hand_coupling_config
    assert generator.chord_track_sampler.model is chord_transition_model
    assert generator.chord_vocabulary is chord_vocabulary
    assert generator.figure_vocabulary is figure_vocabulary
    assert generator.base_duration_distribution is base_duration_distribution
    assert generator.duration_vocabulary is duration_vocabulary
    assert generator.figure_lengths == (2, 3)
