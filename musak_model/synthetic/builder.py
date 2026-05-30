from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.base_durations import BaseDurationDistribution
from musak_model.synthetic.figures import AnchoredFigureVocabulary, FigureVocabulary
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldOverride, AccentFieldSampler
from musak_model.synthetic.processes.chord_track import ChordTrackSampler, ChordTransitionModel
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig, HandCouplingSampler
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveOverride, RegisterCurveSampler
from musak_model.synthetic.substitution import FigureByChordModel, SegmentGenerator, SubstitutionConfig
from musak_model.tokens.duration import DurationVocabulary


def build_segment_generator(
    *,
    substitution_config: SubstitutionConfig,
    register_curve_config: RegisterCurveConfig,
    register_curve_overrides: tuple[RegisterCurveOverride, ...] = (),
    accent_field_config: AccentFieldConfig,
    accent_field_overrides: tuple[AccentFieldOverride, ...] = (),
    hand_coupling_config: HandCouplingConfig,
    chord_transition_model: ChordTransitionModel,
    chord_vocabulary: ChordVocabularyConfig,
    figure_vocabulary: FigureVocabulary,
    anchored_figure_vocabulary: AnchoredFigureVocabulary = AnchoredFigureVocabulary(entries=()),
    figure_by_chord_model: FigureByChordModel = FigureByChordModel(),
    base_duration_distribution: BaseDurationDistribution,
    duration_vocabulary: DurationVocabulary,
    figure_lengths: tuple[int, ...],
) -> SegmentGenerator:
    return SegmentGenerator(
        substitution_config=substitution_config,
        register_curve_sampler=RegisterCurveSampler(config=register_curve_config, overrides=register_curve_overrides),
        accent_field_sampler=AccentFieldSampler(config=accent_field_config, overrides=accent_field_overrides),
        hand_coupling_sampler=HandCouplingSampler(config=hand_coupling_config),
        chord_track_sampler=ChordTrackSampler(model=chord_transition_model),
        chord_vocabulary=chord_vocabulary,
        figure_vocabulary=figure_vocabulary,
        anchored_figure_vocabulary=anchored_figure_vocabulary,
        figure_by_chord_model=figure_by_chord_model,
        base_duration_distribution=base_duration_distribution,
        duration_vocabulary=duration_vocabulary,
        figure_lengths=figure_lengths,
    )
