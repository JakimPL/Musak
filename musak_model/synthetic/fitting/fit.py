from pathlib import Path

from musak_model.harmony.decoding.candidates import spellable_candidates
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.profile.chord.io import read_chord_transitions
from musak_model.n_grams.profile.chord.schema import ChordTransitionCounts, chord_artifact_paths_for_figure_root
from musak_model.n_grams.profile.register.io import read_register_metadata, read_register_statistics
from musak_model.n_grams.profile.register.schema import register_artifact_paths_for_figure_root
from musak_model.n_grams.profile.rhythm.io import read_rhythm_counts
from musak_model.n_grams.profile.rhythm.schema import rhythm_artifact_paths_for_figure_root
from musak_model.synthetic.fitting.accent import fit_accent_overrides_from_rhythm_counts
from musak_model.synthetic.fitting.artifacts import FittedChordTransitions, FittedGeneratorConfig
from musak_model.synthetic.fitting.chord import ChordFitConfig, fit_chord_transition_model
from musak_model.synthetic.fitting.register import fit_register_overrides_from_statistics
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.synthetic.processes.chord_track import functional_transition_model
from musak_model.synthetic.processes.pitch import RegisterCurveConfig
from musak_model.tokens.schema import ScaleType


def fit_generator_config(
    figure_root_directory: Path,
    *,
    register_default: RegisterCurveConfig,
    accent_default: AccentFieldConfig,
    chord_fit: ChordFitConfig,
    chord_vocabulary: ChordVocabularyConfig,
    grid_denominator: int,
) -> FittedGeneratorConfig:
    register_paths = register_artifact_paths_for_figure_root(figure_root_directory)
    metadata = read_register_metadata(register_paths.metadata_path)
    if metadata.arch_basis_count != register_default.arch_basis_count:
        raise ValueError(
            f"register statistics were computed with arch_basis_count={metadata.arch_basis_count}, "
            f"but the register config uses arch_basis_count={register_default.arch_basis_count}"
        )

    register_statistics = read_register_statistics(register_paths.statistics_path)
    rhythm_paths = rhythm_artifact_paths_for_figure_root(figure_root_directory)
    rhythm_counts = read_rhythm_counts(rhythm_paths.counts_path)
    return FittedGeneratorConfig(
        register_overrides=fit_register_overrides_from_statistics(register_statistics, default=register_default),
        accent_overrides=fit_accent_overrides_from_rhythm_counts(
            rhythm_counts, default=accent_default, grid_denominator=grid_denominator
        ),
        chord_transitions=_fit_chord_transitions_from_store(
            figure_root_directory, chord_fit=chord_fit, chord_vocabulary=chord_vocabulary
        ),
    )


def _fit_chord_transitions_from_store(
    figure_root_directory: Path,
    *,
    chord_fit: ChordFitConfig,
    chord_vocabulary: ChordVocabularyConfig,
) -> dict[ScaleType, FittedChordTransitions]:
    chord_paths = chord_artifact_paths_for_figure_root(figure_root_directory)
    if not chord_paths.transitions_path.exists():
        return {}

    return _fit_chord_transitions(
        read_chord_transitions(chord_paths.transitions_path), chord_fit=chord_fit, chord_vocabulary=chord_vocabulary
    )


def _fit_chord_transitions(
    transition_counts: ChordTransitionCounts,
    *,
    chord_fit: ChordFitConfig,
    chord_vocabulary: ChordVocabularyConfig,
) -> dict[ScaleType, FittedChordTransitions]:
    scale_types = {ScaleType(key.scale_type) for key in transition_counts}
    fitted: dict[ScaleType, FittedChordTransitions] = {}
    for scale_type in scale_types:
        chords = tuple(candidate.chord for candidate in spellable_candidates(chord_vocabulary, scale_type=scale_type))
        if not chords:
            continue

        prior = functional_transition_model(
            chords,
            scale_type=scale_type,
            strength=chord_fit.functional_strength,
            self_transition_bias=chord_fit.self_transition_bias,
        )
        model = fit_chord_transition_model(
            transition_counts, scale_type=scale_type, prior=prior, prior_count=chord_fit.prior_count
        )
        fitted[scale_type] = FittedChordTransitions.from_model(model)

    return fitted
