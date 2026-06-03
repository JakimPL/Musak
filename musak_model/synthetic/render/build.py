from __future__ import annotations

from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.inputs import SyntheticInputs
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler
from musak_model.synthetic.processes.density import RhythmicDensityConfig, RhythmicDensitySampler
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.motif import MotifConfig
from musak_model.synthetic.render.renderer import SurfaceRenderer
from musak_model.synthetic.structure.harmony_grammar import HarmonyGrammarConfig, HarmonyGrammarSampler
from musak_model.synthetic.structure.meter import MetricalGrammarConfig, MetricalTreeSampler


def build_surface_renderer(
    inputs: SyntheticInputs,
    *,
    render_config: RenderConfig,
    motif_config: MotifConfig,
    density_config: RhythmicDensityConfig,
) -> SurfaceRenderer:
    chord_vocabulary = ChordVocabularyConfig.load()
    return SurfaceRenderer(
        config=render_config,
        metrical_sampler=MetricalTreeSampler(config=MetricalGrammarConfig.load()),
        harmony_sampler=HarmonyGrammarSampler(config=HarmonyGrammarConfig.load(), vocabulary=chord_vocabulary),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig.load(), overrides=inputs.fitted.register_overrides
        ),
        figure_vocabulary=inputs.figure_vocabulary,
        duration_vocabulary=inputs.duration_vocabulary,
        chord_vocabulary=chord_vocabulary,
        base_duration_distribution=inputs.base_duration_distribution,
        rhythmic_density_sampler=RhythmicDensitySampler(config=density_config),
        accent_field_sampler=AccentFieldSampler(
            config=AccentFieldConfig.load(), overrides=inputs.fitted.accent_overrides
        ),
        grid_denominator=inputs.fitted.grid_denominator,
        motif_config=motif_config,
    )
