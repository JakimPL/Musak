from collections import Counter
from fractions import Fraction
from pathlib import Path

from numpy.random import default_rng

from musak_model.generation.constraints import GenerationConstraints
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.base_durations import BaseDurationDistribution
from musak_model.synthetic.figures import FigureVocabulary
from musak_model.synthetic.harmony.schema import Chord, ChordQuality
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler
from musak_model.synthetic.processes.chord_track import ChordTrackSampler, uniform_transition_model
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig, HandCouplingSampler
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.substitution import GenerationTrace, SegmentGenerator, SubstitutionConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType


def _figure(positions: list[int]) -> FigureNGram:
    onsets = tuple((((position, 0),), Fraction(1)) for position in positions)
    return FigureNGram(onsets=onsets)


def _major_vocabulary() -> FigureVocabulary:
    figure = _figure([0, 2])
    counts = {
        ScaleType.MAJOR: {
            Hand.RIGHT: {2: Counter({figure: 1})},
            Hand.LEFT: {2: Counter({figure: 1})},
        }
    }
    return FigureVocabulary.from_counts(counts)


def _base_durations() -> BaseDurationDistribution:
    return BaseDurationDistribution(
        weights_by_group={
            (ScaleType.MAJOR, Hand.RIGHT, 2): ((Fraction(1, 2), 1),),
            (ScaleType.MAJOR, Hand.LEFT, 2): ((Fraction(1, 2), 1),),
        }
    )


def _generator(duration_vocabulary: DurationVocabulary) -> SegmentGenerator:
    return SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0,
            lambda_harm=0.0,
            lambda_accent=0.0,
            lambda_chord_figure=0.0,
            commonness_bias=1.0,
            max_resample_retries=4,
            monophonic=False,
        ),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig(
                arch_basis_count=3, arch_amplitude=0.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=0.0
            )
        ),
        accent_field_sampler=AccentFieldSampler(
            config=AccentFieldConfig(
                baseline_logit=0.0,
                metric_gain=0.0,
                metric_exponent=1.0,
                envelope_basis_count=3,
                envelope_amplitude=0.0,
                envelope_decay=1.0,
            )
        ),
        hand_coupling_sampler=HandCouplingSampler(
            config=HandCouplingConfig(
                co_activity_strength=0.5, activity_right=1.0, activity_left=1.0, sync_strength=0.0
            )
        ),
        chord_track_sampler=ChordTrackSampler(
            model=uniform_transition_model((Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),))
        ),
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=_major_vocabulary(),
        base_duration_distribution=_base_durations(),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2,),
    )


def test_register_midi_pitch_matches_home_octave_sanity() -> None:
    scale_size = 7

    right = SegmentGenerator._register_midi_pitch(
        anchor=0, scale_size=scale_size, scale_root=0, scale_type=ScaleType.MAJOR, hand=Hand.RIGHT
    )
    left = SegmentGenerator._register_midi_pitch(
        anchor=0, scale_size=scale_size, scale_root=0, scale_type=ScaleType.MAJOR, hand=Hand.LEFT
    )

    assert right == 72
    assert left == 48


def test_trace_has_one_sample_per_hand_per_bar(duration_vocabulary: DurationVocabulary) -> None:
    bar_count = 3
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=bar_count)

    result = _generator(duration_vocabulary).generate(
        bar_count=bar_count,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=1,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(7),
        source_file=Path("synthetic.mxl"),
    )

    trace = result.trace
    assert isinstance(trace, GenerationTrace)
    assert trace.bar_count == bar_count
    assert trace.grid_count_per_bar == 1
    assert len(trace.samples) == bar_count * 2

    for bar_index in range(bar_count):
        bar_samples = [sample for sample in trace.samples if sample.bar_index == bar_index]
        assert {sample.hand for sample in bar_samples} == {Hand.RIGHT, Hand.LEFT}
        for sample in bar_samples:
            assert sample.position == 0
            assert sample.start_in_bars == 1 + bar_index


def test_trace_register_pitch_matches_zero_anchor(duration_vocabulary: DurationVocabulary) -> None:
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1)

    result = _generator(duration_vocabulary).generate(
        bar_count=1,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=1,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(7),
        source_file=Path("synthetic.mxl"),
    )

    by_hand = {sample.hand: sample for sample in result.trace.samples}
    assert by_hand[Hand.RIGHT].register_anchor == 0
    assert by_hand[Hand.RIGHT].register_midi_pitch == 72
    assert by_hand[Hand.LEFT].register_midi_pitch == 48
