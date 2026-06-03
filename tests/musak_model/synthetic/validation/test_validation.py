from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.harmony.diatonic import natural_triad
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.synthetic.render.renderer import RenderedChord
from musak_model.synthetic.validation.adapter import generation_sample
from musak_model.synthetic.validation.config import SyntheticValidationConfig
from musak_model.synthetic.validation.generation import GeneratedSample
from musak_model.synthetic.validation.metrics import validation_metrics
from musak_model.synthetic.validation.options import metric_options
from musak_model.synthetic.validation.synthetic_metrics import synthetic_metrics
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, HandToken, NoteToken, RestToken, ScaleType

# A C-major bar of eight ascending eighth notes (C D E F G A B C'). On-beats (0, 1/4, 1/2, 3/4) land on
# C E G B; off-beats on D F A C'. Under the tonic triad {C, E, G} that is 3/4 chord tones on the beat,
# 1/4 off the beat, and the line walks purely by step.
_SCALE_NOTES = ((1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (1, 1))


def _eighth_note_bar(duration_vocabulary: DurationVocabulary) -> Segment:
    eighth = duration_vocabulary.require_duration_id(Fraction(1, 8))
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    tokens = [
        HandToken(hand="right"),
        *[
            NoteToken(degree=degree, accidental=0, octave_offset=octave, duration_id=eighth)
            for degree, octave in _SCALE_NOTES
        ],
        HandToken(hand="left"),
        RestToken(duration_id=whole),
        BarToken(),
        EndToken(),
    ]
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("validation-test"),
            difficulty_level=None,
        ),
    )


def _generated_sample(duration_vocabulary: DurationVocabulary) -> GeneratedSample:
    return GeneratedSample(
        scale_type=ScaleType.MAJOR,
        seed=0,
        segment=_eighth_note_bar(duration_vocabulary),
        chords=(RenderedChord(offset=Fraction(0), duration=Fraction(1), chord=natural_triad(ScaleType.MAJOR, 1)),),
        render_error=None,
    )


def _config() -> SyntheticValidationConfig:
    return SyntheticValidationConfig.load().model_copy(update={"bar_count": 1})


def _options() -> object:
    return metric_options(_config(), ScaleType.MAJOR)


def test_synthetic_metrics_reward_chord_tones_on_strong_beats(duration_vocabulary: DurationVocabulary) -> None:
    metrics = synthetic_metrics(
        [_generated_sample(duration_vocabulary)],
        options=_options(),
        chord_vocabulary=ChordVocabularyConfig.load(),
        duration_vocabulary=duration_vocabulary,
    )

    assert metrics["generation/synthetic/strong_beat_chord_tone_fraction"] == 0.75
    assert metrics["generation/synthetic/weak_beat_chord_tone_fraction"] == 0.25
    assert metrics["generation/synthetic/chord_tone_strong_weak_gap"] == 0.5
    assert metrics["generation/synthetic/stepwise_fraction"] == 1.0  # the scale walks by step
    assert metrics["generation/synthetic/chord_onset_fraction"] == 0.0  # single-note onsets only


def test_adapter_builds_a_constraint_valid_sample(duration_vocabulary: DurationVocabulary) -> None:
    sample = generation_sample(
        _eighth_note_bar(duration_vocabulary), options=_options(), duration_vocabulary=duration_vocabulary
    )

    assert sample.reached_end
    assert sample.completed_bars == 1
    assert not sample.constraint_report.failed
    assert sample.diagnostics is not None
    assert sample.decode_error is None


def test_validation_metrics_aggregate_and_count_render_errors(duration_vocabulary: DurationVocabulary) -> None:
    samples = {
        ScaleType.MAJOR: [
            _generated_sample(duration_vocabulary),
            GeneratedSample(scale_type=ScaleType.MAJOR, seed=1, segment=None, chords=(), render_error="boom"),
        ]
    }

    metrics = validation_metrics(
        samples,
        config=_config(),
        artifacts=None,
        chord_vocabulary=ChordVocabularyConfig.load(),
        duration_vocabulary=duration_vocabulary,
        rhythm_config=NGramAnalysisConfig.load().rhythm_analysis,
    )

    assert metrics["generation/major/rate/render_error"] == 0.5
    assert metrics["generation/major/count/rendered"] == 1.0
    assert metrics["generation/overall/count/samples"] == 1.0
    assert "generation/major/synthetic/strong_beat_chord_tone_fraction" in metrics
    # No reference artifacts → figure/rhythm fidelity metrics are skipped.
    assert not any("identity_total_variation_distance" in name for name in metrics)
