from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.profile.io import read_figure_counts
from musak_model.synthetic.base_durations import load_base_duration_distribution
from musak_model.synthetic.builder import build_segment_generator
from musak_model.synthetic.calibration.config import CalibrationConfig
from musak_model.synthetic.figures import load_figure_vocabulary, resolve_figure_counts_path
from musak_model.synthetic.harmony.decoding.candidates import spellable_candidates
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.synthetic.processes.chord_track import uniform_transition_model
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig
from musak_model.synthetic.processes.pitch import RegisterCurveConfig
from musak_model.synthetic.substitution import SegmentGenerator, SubstitutionConfig
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary


def build_calibration_generator(config: CalibrationConfig) -> SegmentGenerator:
    chord_vocabulary = ChordVocabularyConfig.load()
    chords = tuple(
        candidate.chord for candidate in spellable_candidates(chord_vocabulary, scale_type=config.scale_type)
    )
    return build_segment_generator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0,
            lambda_harm=0.0,
            lambda_accent=0.0,
            commonness_bias=config.commonness_bias,
            max_resample_retries=config.max_resample_retries,
            monophonic=False,
        ),
        register_curve_config=RegisterCurveConfig.load(),
        accent_field_config=AccentFieldConfig.load(),
        hand_coupling_config=HandCouplingConfig.load(),
        chord_transition_model=uniform_transition_model(chords, self_transition_bias=config.self_transition_bias),
        chord_vocabulary=chord_vocabulary,
        figure_vocabulary=load_figure_vocabulary(config.figure_root),
        base_duration_distribution=load_base_duration_distribution(config.figure_root),
        duration_vocabulary=DurationVocabulary(TokenizationConfig.load()),
        figure_lengths=config.figure_lengths,
    )


def load_reference_counts(config: CalibrationConfig) -> FigureNGramCountsByScale:
    return read_figure_counts(resolve_figure_counts_path(config.figure_root))
