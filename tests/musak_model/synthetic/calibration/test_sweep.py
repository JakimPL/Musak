import csv
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.base_durations import BaseDurationDistribution
from musak_model.synthetic.calibration.config import CalibrationConfig
from musak_model.synthetic.calibration.results import write_sweep_results
from musak_model.synthetic.calibration.sweep import run_sweep
from musak_model.synthetic.figures import FigureVocabulary
from musak_model.synthetic.harmony.schema import Chord, ChordQuality
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler
from musak_model.synthetic.processes.chord_track import ChordTrackSampler, uniform_transition_model
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig, HandCouplingSampler
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.substitution import SegmentGenerator, SubstitutionConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType


def _figure(positions: list[int]) -> FigureNGram:
    return FigureNGram(onsets=tuple((((position, 0),), Fraction(1)) for position in positions))


def _generator(duration_vocabulary: DurationVocabulary) -> SegmentGenerator:
    vocabulary = FigureVocabulary.from_counts(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {2: Counter({_figure([0, 1]): 1})},
                Hand.LEFT: {2: Counter({_figure([0, 1]): 1})},
            }
        }
    )
    base_durations = BaseDurationDistribution(
        weights_by_group={
            (ScaleType.MAJOR, Hand.RIGHT, 2): ((Fraction(1, 4), 1),),
            (ScaleType.MAJOR, Hand.LEFT, 2): ((Fraction(1, 4), 1),),
        }
    )
    return SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0, lambda_harm=0.0, lambda_accent=0.0, commonness_bias=1.0, max_resample_retries=4
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
            config=HandCouplingConfig(co_activity_strength=0.5, activity_right=1.0, activity_left=1.0)
        ),
        chord_track_sampler=ChordTrackSampler(
            model=uniform_transition_model((Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),))
        ),
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=vocabulary,
        base_duration_distribution=base_durations,
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2,),
    )


def _reference_counts() -> FigureNGramCountsByScale:
    return {ScaleType.MAJOR: {Hand.RIGHT: {2: Counter({_figure([0, 1]): 4})}}}


def _config(
    *,
    lambda_curve: tuple[float, ...] = (0.0, 1.0),
    lambda_harm: tuple[float, ...] = (0.0, 1.0),
    lambda_accent: tuple[float, ...] = (0.0, 1.0),
    samples_per_config: int = 4,
) -> CalibrationConfig:
    return CalibrationConfig(
        figure_root=Path("unused"),
        output_path=Path("unused.tsv"),
        scale_type=ScaleType.MAJOR,
        scale_root=0,
        time_numerator=4,
        time_denominator=4,
        bar_count=2,
        samples_per_config=samples_per_config,
        min_n=2,
        max_n=2,
        self_transition_bias=0.25,
        commonness_bias=1.0,
        max_resample_retries=4,
        seed=0,
        lambda_curve=lambda_curve,
        lambda_harm=lambda_harm,
        lambda_accent=lambda_accent,
    )


def test_run_sweep_covers_full_grid(duration_vocabulary: DurationVocabulary) -> None:
    results = run_sweep(
        generator=_generator(duration_vocabulary),
        reference_counts=_reference_counts(),
        config=_config(),
    )

    assert len(results) == 8
    assert {(result.lambda_curve, result.lambda_harm, result.lambda_accent) for result in results} == {
        (lambda_curve, lambda_harm, lambda_accent)
        for lambda_curve in (0.0, 1.0)
        for lambda_harm in (0.0, 1.0)
        for lambda_accent in (0.0, 1.0)
    }
    for result in results:
        assert result.distribution_groups == 1
        assert result.mean_total_variation_distance is not None
        assert 0.0 <= result.mean_total_variation_distance <= 1.0


def test_run_sweep_is_deterministic(duration_vocabulary: DurationVocabulary) -> None:
    config = _config(lambda_curve=(0.0,), lambda_harm=(0.0,), lambda_accent=(0.0,))

    first = run_sweep(generator=_generator(duration_vocabulary), reference_counts=_reference_counts(), config=config)
    second = run_sweep(generator=_generator(duration_vocabulary), reference_counts=_reference_counts(), config=config)

    assert first == second


def test_calibration_config_rejects_inverted_n_range() -> None:
    with pytest.raises(ValueError, match="max_n"):
        _config_with_n_range(min_n=3, max_n=2)


def test_calibration_config_rejects_empty_lambda_grid() -> None:
    with pytest.raises(ValueError, match="lambda_curve"):
        _config(lambda_curve=())


def test_write_sweep_results_writes_sortable_csv(tmp_path: Path, duration_vocabulary: DurationVocabulary) -> None:
    results = run_sweep(
        generator=_generator(duration_vocabulary),
        reference_counts=_reference_counts(),
        config=_config(),
    )
    output_path = tmp_path / "sweep.csv"

    write_sweep_results(results, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == [
        "lambda_curve",
        "lambda_harm",
        "lambda_accent",
        "distribution_groups",
        "mean_total_variation_distance",
    ]
    assert len(rows) == 9
    assert all(len(row) == 5 for row in rows)


def _config_with_n_range(*, min_n: int, max_n: int) -> CalibrationConfig:
    return CalibrationConfig(
        figure_root=Path("unused"),
        output_path=Path("unused.tsv"),
        scale_type=ScaleType.MAJOR,
        scale_root=0,
        time_numerator=4,
        time_denominator=4,
        bar_count=2,
        samples_per_config=4,
        min_n=min_n,
        max_n=max_n,
        self_transition_bias=0.25,
        commonness_bias=1.0,
        max_resample_retries=4,
        seed=0,
        lambda_curve=(0.0,),
        lambda_harm=(0.0,),
        lambda_accent=(0.0,),
    )
