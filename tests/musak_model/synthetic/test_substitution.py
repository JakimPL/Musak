from collections import Counter
from fractions import Fraction
from pathlib import Path

from numpy.random import default_rng

from musak_model.data.schema import Segment
from musak_model.generation.constraints import (
    GenerationConstraints,
    GenerationConstraintState,
)
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.base_durations import BaseDurationDistribution
from musak_model.synthetic.figures import FigureVocabulary
from musak_model.synthetic.harmony.expansion import chord_pitch_class_set
from musak_model.synthetic.harmony.schema import Chord, ChordQuality
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler
from musak_model.synthetic.processes.chord_track import (
    ChordTrackSampler,
    ChordTransitionModel,
    uniform_transition_model,
)
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig, HandCouplingSampler
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.substitution import (
    SegmentGenerationResult,
    SegmentGenerator,
    SubstitutionConfig,
    accent_fit,
    anchor_figure_to_tokens,
    figure_net_contour,
    harm_fit,
    is_monorhythmic,
    monorhythmic_entries,
    sample_substituted_figure,
    slope_fit,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, Hand, HandToken, NoteToken, RestToken, ScaleType, Token


def _figure(positions: list[int], *, durations: list[Fraction] | None = None) -> FigureNGram:
    actual_durations = durations or [Fraction(1)] * len(positions)
    onsets = tuple((((position, 0),), duration) for position, duration in zip(positions, actual_durations, strict=True))
    return FigureNGram(onsets=onsets)


def _major_vocabulary(figures_by_n_per_hand: dict[Hand, dict[int, list[FigureNGram]]]) -> FigureVocabulary:
    counts = {
        ScaleType.MAJOR: {
            hand: {figure_length: Counter({figure: 1 for figure in figures}) for figure_length, figures in by_n.items()}
            for hand, by_n in figures_by_n_per_hand.items()
        }
    }
    return FigureVocabulary.from_counts(counts)


def _flat_accent_field_sampler() -> AccentFieldSampler:
    return AccentFieldSampler(
        config=AccentFieldConfig(
            baseline_logit=0.0,
            metric_gain=0.0,
            metric_exponent=1.0,
            envelope_basis_count=3,
            envelope_amplitude=0.0,
            envelope_decay=1.0,
        )
    )


def _always_onset_accent_field_sampler() -> AccentFieldSampler:
    return AccentFieldSampler(
        config=AccentFieldConfig(
            baseline_logit=40.0,
            metric_gain=0.0,
            metric_exponent=1.0,
            envelope_basis_count=3,
            envelope_amplitude=0.0,
            envelope_decay=1.0,
        )
    )


def _always_active_hand_coupling_sampler() -> HandCouplingSampler:
    return HandCouplingSampler(
        config=HandCouplingConfig(co_activity_strength=0.5, activity_right=1.0, activity_left=1.0, sync_strength=0.0)
    )


def _base_durations(bases_by_group: dict[tuple[Hand, int], dict[Fraction, int]]) -> BaseDurationDistribution:
    return BaseDurationDistribution(
        weights_by_group={
            (ScaleType.MAJOR, hand, figure_length): tuple(sorted(bases.items()))
            for (hand, figure_length), bases in bases_by_group.items()
        }
    )


def _uniform_base_durations(*, figure_length: int, base_duration: Fraction) -> BaseDurationDistribution:
    return _base_durations(
        {
            (Hand.RIGHT, figure_length): {base_duration: 1},
            (Hand.LEFT, figure_length): {base_duration: 1},
        }
    )


def _tokens_under_hand(tokens: list[Token], hand: Hand) -> list[Token]:
    collected: list[Token] = []
    current_hand: Hand | None = None
    for token in tokens:
        if isinstance(token, HandToken):
            current_hand = token.hand
        elif isinstance(token, BarToken):
            current_hand = None
        elif current_hand == hand:
            collected.append(token)

    return collected


def test_is_monorhythmic_detects_equal_normalized_durations() -> None:
    assert is_monorhythmic(_figure([0, 1]))
    assert not is_monorhythmic(_figure([0, 1], durations=[Fraction(1), Fraction(2)]))


def test_figure_net_contour_uses_last_onset_min_position() -> None:
    assert figure_net_contour(_figure([0, 2])) == 2
    assert figure_net_contour(_figure([0, -1, 0])) == 0
    assert figure_net_contour(_figure([0, 2, 4])) == 4


def test_slope_fit_rewards_matching_net_contour() -> None:
    figure = _figure([0, 2])

    assert slope_fit(figure=figure, target_slope=2) == 0.0
    assert slope_fit(figure=figure, target_slope=0) == -2.0
    assert slope_fit(figure=figure, target_slope=4) == -2.0


def test_harm_fit_counts_chord_tone_fraction() -> None:
    chord_pcs = frozenset({0, 4, 7})

    all_chord_tones = _figure([0, 2])
    none_chord_tones = _figure([1, 5])
    mixed = _figure([0, 1])

    assert harm_fit(figure=all_chord_tones, anchor=0, scale_type=ScaleType.MAJOR, chord_pitch_classes=chord_pcs) == 1.0
    assert harm_fit(figure=none_chord_tones, anchor=0, scale_type=ScaleType.MAJOR, chord_pitch_classes=chord_pcs) == 0.0
    assert harm_fit(figure=mixed, anchor=0, scale_type=ScaleType.MAJOR, chord_pitch_classes=chord_pcs) == 0.5


def test_accent_fit_scales_with_envelope_value() -> None:
    figure = _figure([0, 2])

    assert _rhythm_accent_fit(figure, envelope_value=0.0) == 0.0
    half = _rhythm_accent_fit(figure, envelope_value=0.5)
    full = _rhythm_accent_fit(figure, envelope_value=1.0)
    assert full == 2 * half


def test_accent_fit_prefers_front_loaded_over_uniform() -> None:
    uniform = _figure([0, 1], durations=[Fraction(1), Fraction(1)])
    front_loaded = _figure([0, 1], durations=[Fraction(2), Fraction(1)])
    back_loaded = _figure([0, 1], durations=[Fraction(1), Fraction(2)])

    uniform_score = _rhythm_accent_fit(uniform, envelope_value=1.0)
    front_score = _rhythm_accent_fit(front_loaded, envelope_value=1.0)
    back_score = _rhythm_accent_fit(back_loaded, envelope_value=1.0)

    assert front_score > uniform_score > back_score


def test_accent_fit_handles_single_onset_figure() -> None:
    assert _rhythm_accent_fit(_figure([0]), envelope_value=0.7) == 0.7


def test_accent_fit_rewards_chord_tone_on_strong_onset() -> None:
    chord_pcs = frozenset({0, 4, 7})
    chord_tone_on_strong = _figure([0, 1])
    chord_tone_on_weak = _figure([1, 0])

    strong_score = accent_fit(
        figure=chord_tone_on_strong,
        anchor=0,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=chord_pcs,
        envelope_value=1.0,
    )
    weak_score = accent_fit(
        figure=chord_tone_on_weak,
        anchor=0,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=chord_pcs,
        envelope_value=1.0,
    )

    assert strong_score > weak_score


def _rhythm_accent_fit(figure: FigureNGram, *, envelope_value: float) -> float:
    return accent_fit(
        figure=figure,
        anchor=0,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=frozenset(),
        envelope_value=envelope_value,
    )


def test_chord_pitch_class_set_matches_expansion() -> None:
    vocabulary = ChordVocabularyConfig.load()
    tonic = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)

    assert chord_pitch_class_set(tonic, scale_type=ScaleType.MAJOR, vocabulary=vocabulary) == frozenset({0, 4, 7})


def test_anchor_figure_to_tokens_emits_absolute_degrees_and_durations(
    duration_vocabulary: DurationVocabulary,
) -> None:
    figure = _figure([0, 2])

    tokens = anchor_figure_to_tokens(
        figure=figure,
        anchor=0,
        base_duration=Fraction(1, 2),
        scale_type=ScaleType.MAJOR,
        duration_vocabulary=duration_vocabulary,
    )

    assert [type(token).__name__ for token in tokens] == ["NoteToken", "NoteToken"]
    first, second = tokens[0], tokens[1]
    assert isinstance(first, NoteToken) and isinstance(second, NoteToken)
    assert (first.degree, first.octave_offset) == (1, 0)
    assert (second.degree, second.octave_offset) == (3, 0)
    assert duration_vocabulary.id_to_fraction(first.duration_id) == Fraction(1, 2)


def test_sample_substituted_figure_is_deterministic_for_a_given_seed() -> None:
    entries = monorhythmic_entries(
        _major_vocabulary({Hand.RIGHT: {2: [_figure([0, 2]), _figure([0, -1])]}}),
        scale_type=ScaleType.MAJOR,
        hand=Hand.RIGHT,
        figure_length=2,
    )
    config = SubstitutionConfig(
        lambda_curve=1.0, lambda_harm=1.0, lambda_accent=0.0, commonness_bias=1.0, max_resample_retries=4
    )

    first = sample_substituted_figure(
        entries=entries,
        anchor=0,
        target_slope=2,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=frozenset({0, 4, 7}),
        envelope_value=0.0,
        config=config,
        rng=default_rng(11),
    )
    second = sample_substituted_figure(
        entries=entries,
        anchor=0,
        target_slope=2,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=frozenset({0, 4, 7}),
        envelope_value=0.0,
        config=config,
        rng=default_rng(11),
    )

    assert first == second


def test_high_lambda_curve_selects_the_slope_matching_figure() -> None:
    ascending = _figure([0, 2])
    descending = _figure([0, -2])
    entries = monorhythmic_entries(
        _major_vocabulary({Hand.RIGHT: {2: [ascending, descending]}}),
        scale_type=ScaleType.MAJOR,
        hand=Hand.RIGHT,
        figure_length=2,
    )
    config = SubstitutionConfig(
        lambda_curve=50.0, lambda_harm=0.0, lambda_accent=0.0, commonness_bias=0.0, max_resample_retries=4
    )

    chosen = sample_substituted_figure(
        entries=entries,
        anchor=0,
        target_slope=2,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=frozenset({0, 4, 7}),
        envelope_value=0.0,
        config=config,
        rng=default_rng(0),
    )

    assert chosen.figure == ascending


def test_high_lambda_accent_selects_front_loaded_figure_under_high_envelope() -> None:
    uniform = _figure([0, 1], durations=[Fraction(1), Fraction(1)])
    front_loaded = _figure([0, 1], durations=[Fraction(2), Fraction(1)])
    vocabulary = _major_vocabulary({Hand.RIGHT: {2: [uniform, front_loaded]}})
    entries = vocabulary.filter(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, n=2).entries
    config = SubstitutionConfig(
        lambda_curve=0.0, lambda_harm=0.0, lambda_accent=50.0, commonness_bias=0.0, max_resample_retries=4
    )

    chosen = sample_substituted_figure(
        entries=entries,
        anchor=0,
        target_slope=0,
        scale_type=ScaleType.MAJOR,
        chord_pitch_classes=frozenset({0, 4, 7}),
        envelope_value=1.0,
        config=config,
        rng=default_rng(0),
    )

    assert chosen.figure == front_loaded


def test_segment_generator_produces_constraint_valid_segment(
    duration_vocabulary: DurationVocabulary,
) -> None:
    vocabulary = _major_vocabulary(
        {
            Hand.RIGHT: {2: [_figure([0, 2])]},
            Hand.LEFT: {2: [_figure([0, 2])]},
        }
    )
    chord_track_sampler = ChordTrackSampler(
        model=uniform_transition_model((Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),))
    )
    register_curve_sampler = RegisterCurveSampler(
        config=RegisterCurveConfig(
            arch_basis_count=3,
            arch_amplitude=0.0,
            arch_decay=1.0,
            ou_theta=0.5,
            ou_sigma=0.0,
        )
    )
    generator = SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0, lambda_harm=0.0, lambda_accent=0.0, commonness_bias=1.0, max_resample_retries=4
        ),
        register_curve_sampler=register_curve_sampler,
        accent_field_sampler=_flat_accent_field_sampler(),
        hand_coupling_sampler=_always_active_hand_coupling_sampler(),
        chord_track_sampler=chord_track_sampler,
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=vocabulary,
        base_duration_distribution=_uniform_base_durations(figure_length=2, base_duration=Fraction(1, 2)),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2,),
    )
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=2)

    segment = generator.generate(
        bar_count=2,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=1,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(7),
        source_file=Path("synthetic.mxl"),
    ).segment

    assert isinstance(segment, Segment)
    state = GenerationConstraintState(constraints=constraints)
    for token in segment.tokens:
        state = state.apply(token, duration_vocabulary=duration_vocabulary)
    assert state.ended
    assert state.bar_index == 2


def test_sub_bar_chord_resolution_conditions_each_half_bar(
    duration_vocabulary: DurationVocabulary,
) -> None:
    c_major = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)
    g_major = Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR)
    chord_track_sampler = ChordTrackSampler(
        model=ChordTransitionModel(
            initial_distribution={c_major: 1.0, g_major: 0.0},
            transitions={
                c_major: {c_major: 0.0, g_major: 1.0},
                g_major: {c_major: 0.0, g_major: 1.0},
            },
        )
    )
    tonic_note = _figure([0])
    dominant_note = _figure([1])
    generator = SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0, lambda_harm=50.0, lambda_accent=0.0, commonness_bias=0.0, max_resample_retries=4
        ),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig(
                arch_basis_count=3, arch_amplitude=0.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=0.0
            )
        ),
        accent_field_sampler=_always_onset_accent_field_sampler(),
        hand_coupling_sampler=HandCouplingSampler(
            config=HandCouplingConfig(
                co_activity_strength=0.5, activity_right=1.0, activity_left=0.0, sync_strength=0.0
            )
        ),
        chord_track_sampler=chord_track_sampler,
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=_major_vocabulary({Hand.RIGHT: {1: [tonic_note, dominant_note]}}),
        base_duration_distribution=_uniform_base_durations(figure_length=1, base_duration=Fraction(1, 2)),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(1,),
    )
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1)

    segment = generator.generate(
        bar_count=1,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=2,
        chord_resolution=2,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(0),
        source_file=Path("synthetic.mxl"),
    ).segment

    right_notes = [token for token in _tokens_under_hand(segment.tokens, Hand.RIGHT) if isinstance(token, NoteToken)]
    assert [note.degree for note in right_notes] == [1, 2]


def test_silenced_hand_emits_rest_filled_bars(
    duration_vocabulary: DurationVocabulary,
) -> None:
    vocabulary = _major_vocabulary(
        {
            Hand.RIGHT: {2: [_figure([0, 2])]},
            Hand.LEFT: {2: [_figure([0, 2])]},
        }
    )
    generator = SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0, lambda_harm=0.0, lambda_accent=0.0, commonness_bias=1.0, max_resample_retries=4
        ),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig(
                arch_basis_count=3, arch_amplitude=0.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=0.0
            )
        ),
        accent_field_sampler=_always_onset_accent_field_sampler(),
        hand_coupling_sampler=HandCouplingSampler(
            config=HandCouplingConfig(
                co_activity_strength=0.5, activity_right=1.0, activity_left=0.0, sync_strength=0.0
            )
        ),
        chord_track_sampler=ChordTrackSampler(
            model=uniform_transition_model((Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),))
        ),
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=vocabulary,
        base_duration_distribution=_uniform_base_durations(figure_length=2, base_duration=Fraction(1, 2)),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2,),
    )
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=2)

    segment = generator.generate(
        bar_count=2,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=1,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(0),
        source_file=Path("synthetic.mxl"),
    ).segment

    whole_rest_id = duration_vocabulary.require_duration_id(Fraction(1))
    left_hand_tokens = _tokens_under_hand(segment.tokens, Hand.LEFT)
    right_hand_tokens = _tokens_under_hand(segment.tokens, Hand.RIGHT)
    assert left_hand_tokens and all(token == RestToken(duration_id=whole_rest_id) for token in left_hand_tokens)
    assert all(isinstance(token, NoteToken) for token in right_hand_tokens)

    state = GenerationConstraintState(constraints=constraints)
    for token in segment.tokens:
        state = state.apply(token, duration_vocabulary=duration_vocabulary)
    assert state.ended
    assert state.bar_index == 2


def test_segment_generator_is_deterministic_for_a_given_seed(
    duration_vocabulary: DurationVocabulary,
) -> None:
    vocabulary = _major_vocabulary(
        {
            Hand.RIGHT: {2: [_figure([0, 2]), _figure([0, -1])]},
            Hand.LEFT: {2: [_figure([0, 2])]},
        }
    )
    generator = SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0, lambda_harm=0.0, lambda_accent=0.0, commonness_bias=1.0, max_resample_retries=4
        ),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig(
                arch_basis_count=3,
                arch_amplitude=2.0,
                arch_decay=1.0,
                ou_theta=0.3,
                ou_sigma=0.5,
            )
        ),
        accent_field_sampler=_flat_accent_field_sampler(),
        hand_coupling_sampler=_always_active_hand_coupling_sampler(),
        chord_track_sampler=ChordTrackSampler(
            model=uniform_transition_model((Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),))
        ),
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=vocabulary,
        base_duration_distribution=_uniform_base_durations(figure_length=2, base_duration=Fraction(1, 2)),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2,),
    )
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=3)

    first = generator.generate(
        bar_count=3,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=1,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(123),
        source_file=Path("synthetic.mxl"),
    ).segment
    second = generator.generate(
        bar_count=3,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=1,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(123),
        source_file=Path("synthetic.mxl"),
    ).segment

    assert first.tokens == second.tokens


def test_segment_generator_places_multiple_figures_per_bar(
    duration_vocabulary: DurationVocabulary,
) -> None:
    vocabulary = _major_vocabulary(
        {
            Hand.RIGHT: {2: [_figure([0, 1])]},
            Hand.LEFT: {2: [_figure([0, 1])]},
        }
    )
    generator = SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0, lambda_harm=0.0, lambda_accent=0.0, commonness_bias=1.0, max_resample_retries=4
        ),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig(
                arch_basis_count=3, arch_amplitude=0.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=0.0
            )
        ),
        accent_field_sampler=_always_onset_accent_field_sampler(),
        hand_coupling_sampler=_always_active_hand_coupling_sampler(),
        chord_track_sampler=ChordTrackSampler(
            model=uniform_transition_model((Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),))
        ),
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=vocabulary,
        base_duration_distribution=_uniform_base_durations(figure_length=2, base_duration=Fraction(1, 8)),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2,),
    )
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1)

    segment = generator.generate(
        bar_count=1,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=4,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(3),
        source_file=Path("synthetic.mxl"),
    ).segment

    eighth = duration_vocabulary.require_duration_id(Fraction(1, 8))
    right_hand_tokens = _tokens_under_hand(segment.tokens, Hand.RIGHT)
    assert all(isinstance(token, NoteToken) and token.duration_id == eighth for token in right_hand_tokens)
    assert len(right_hand_tokens) == 8

    state = GenerationConstraintState(constraints=constraints)
    for token in segment.tokens:
        state = state.apply(token, duration_vocabulary=duration_vocabulary)
    assert state.ended
    assert state.bar_index == 1


def test_segment_generator_rests_the_trailing_gap_when_no_figure_fits(
    duration_vocabulary: DurationVocabulary,
) -> None:
    vocabulary = _major_vocabulary(
        {
            Hand.RIGHT: {2: [_figure([0, 1])]},
            Hand.LEFT: {2: [_figure([0, 1])]},
        }
    )
    generator = SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0, lambda_harm=0.0, lambda_accent=0.0, commonness_bias=1.0, max_resample_retries=4
        ),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig(
                arch_basis_count=3, arch_amplitude=0.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=0.0
            )
        ),
        accent_field_sampler=_always_onset_accent_field_sampler(),
        hand_coupling_sampler=_always_active_hand_coupling_sampler(),
        chord_track_sampler=ChordTrackSampler(
            model=uniform_transition_model((Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),))
        ),
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=vocabulary,
        base_duration_distribution=_uniform_base_durations(figure_length=2, base_duration=Fraction(3, 8)),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2,),
    )
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1)

    segment = generator.generate(
        bar_count=1,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=1,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(1),
        source_file=Path("synthetic.mxl"),
    ).segment

    right_hand_tokens = _tokens_under_hand(segment.tokens, Hand.RIGHT)
    assert any(isinstance(token, NoteToken) for token in right_hand_tokens)
    assert any(isinstance(token, RestToken) for token in right_hand_tokens)

    state = GenerationConstraintState(constraints=constraints)
    for token in segment.tokens:
        state = state.apply(token, duration_vocabulary=duration_vocabulary)
    assert state.ended
    assert state.bar_index == 1


def _grid_generator(
    duration_vocabulary: DurationVocabulary,
    *,
    accent_field_sampler: AccentFieldSampler,
    hand_coupling_sampler: HandCouplingSampler,
    register_curve_sampler: RegisterCurveSampler,
    base_duration: Fraction,
    figure: FigureNGram,
) -> SegmentGenerator:
    vocabulary = _major_vocabulary(
        {
            Hand.RIGHT: {len(figure.onsets): [figure]},
            Hand.LEFT: {len(figure.onsets): [figure]},
        }
    )
    return SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0, lambda_harm=0.0, lambda_accent=0.0, commonness_bias=1.0, max_resample_retries=4
        ),
        register_curve_sampler=register_curve_sampler,
        accent_field_sampler=accent_field_sampler,
        hand_coupling_sampler=hand_coupling_sampler,
        chord_track_sampler=ChordTrackSampler(
            model=uniform_transition_model((Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),))
        ),
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=vocabulary,
        base_duration_distribution=_uniform_base_durations(
            figure_length=len(figure.onsets), base_duration=base_duration
        ),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(len(figure.onsets),),
    )


def _flat_register_curve_sampler() -> RegisterCurveSampler:
    return RegisterCurveSampler(
        config=RegisterCurveConfig(arch_basis_count=3, arch_amplitude=0.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=0.0)
    )


def test_hand_rests_mid_bar_when_a_sub_bar_cell_does_not_fire(
    duration_vocabulary: DurationVocabulary,
) -> None:
    # A quarter-note grid; some cells fire and some do not, so a single bar mixes figures and rests.
    generator = _grid_generator(
        duration_vocabulary,
        accent_field_sampler=_flat_accent_field_sampler(),
        hand_coupling_sampler=_always_active_hand_coupling_sampler(),
        register_curve_sampler=_flat_register_curve_sampler(),
        base_duration=Fraction(1, 8),
        figure=_figure([0, 1]),
    )
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1)

    segment = generator.generate(
        bar_count=1,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=4,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(2),
        source_file=Path("synthetic.mxl"),
    ).segment

    right_hand_tokens = _tokens_under_hand(segment.tokens, Hand.RIGHT)
    assert any(isinstance(token, NoteToken) for token in right_hand_tokens)
    assert any(isinstance(token, RestToken) for token in right_hand_tokens)

    state = GenerationConstraintState(constraints=constraints)
    for token in segment.tokens:
        state = state.apply(token, duration_vocabulary=duration_vocabulary)
    assert state.ended
    assert state.bar_index == 1


def test_register_anchor_varies_across_cells_within_a_bar(
    duration_vocabulary: DurationVocabulary,
) -> None:
    register_curve_sampler = RegisterCurveSampler(
        config=RegisterCurveConfig(arch_basis_count=3, arch_amplitude=6.0, arch_decay=1.0, ou_theta=0.3, ou_sigma=1.0)
    )
    result = _grid_generator(
        duration_vocabulary,
        accent_field_sampler=_always_onset_accent_field_sampler(),
        hand_coupling_sampler=_always_active_hand_coupling_sampler(),
        register_curve_sampler=register_curve_sampler,
        base_duration=Fraction(1, 8),
        figure=_figure([0, 1]),
    ).generate(
        bar_count=1,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=4,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        rng=default_rng(5),
        source_file=Path("synthetic.mxl"),
    )

    right_anchors = {
        sample.register_anchor for sample in result.trace.samples if sample.hand == Hand.RIGHT and sample.bar_index == 0
    }
    assert len(right_anchors) > 1


def test_accent_weight_varies_across_cells_within_a_bar(
    duration_vocabulary: DurationVocabulary,
) -> None:
    accent_field_sampler = AccentFieldSampler(
        config=AccentFieldConfig(
            baseline_logit=0.0,
            metric_gain=3.0,
            metric_exponent=1.0,
            envelope_basis_count=3,
            envelope_amplitude=1.0,
            envelope_decay=1.0,
        )
    )
    result = _grid_generator(
        duration_vocabulary,
        accent_field_sampler=accent_field_sampler,
        hand_coupling_sampler=_always_active_hand_coupling_sampler(),
        register_curve_sampler=_flat_register_curve_sampler(),
        base_duration=Fraction(1, 8),
        figure=_figure([0, 1]),
    ).generate(
        bar_count=1,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=4,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        rng=default_rng(5),
        source_file=Path("synthetic.mxl"),
    )

    right_weights = {
        sample.accent_weight for sample in result.trace.samples if sample.hand == Hand.RIGHT and sample.bar_index == 0
    }
    assert len(right_weights) > 1


def test_a_single_figure_can_span_multiple_cells(
    duration_vocabulary: DurationVocabulary,
) -> None:
    # A four-onset figure at base duration 1/4 spans a whole 4/4 bar (four quarter cells) from one onset.
    generator = _grid_generator(
        duration_vocabulary,
        accent_field_sampler=_always_onset_accent_field_sampler(),
        hand_coupling_sampler=_always_active_hand_coupling_sampler(),
        register_curve_sampler=_flat_register_curve_sampler(),
        base_duration=Fraction(1, 4),
        figure=_figure([0, 1, 2, 3]),
    )
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1)

    segment = generator.generate(
        bar_count=1,
        time_numerator=4,
        time_denominator=4,
        grid_count_per_bar=4,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(0),
        source_file=Path("synthetic.mxl"),
    ).segment

    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    right_hand_tokens = _tokens_under_hand(segment.tokens, Hand.RIGHT)
    assert len(right_hand_tokens) == 4
    assert all(isinstance(token, NoteToken) and token.duration_id == quarter for token in right_hand_tokens)


def test_segment_generator_seed_determinism_covers_tokens_and_trace(
    duration_vocabulary: DurationVocabulary,
) -> None:
    register_curve_sampler = RegisterCurveSampler(
        config=RegisterCurveConfig(arch_basis_count=3, arch_amplitude=2.0, arch_decay=1.0, ou_theta=0.3, ou_sigma=0.5)
    )
    accent_field_sampler = AccentFieldSampler(
        config=AccentFieldConfig(
            baseline_logit=0.5,
            metric_gain=2.0,
            metric_exponent=1.0,
            envelope_basis_count=3,
            envelope_amplitude=0.5,
            envelope_decay=1.0,
        )
    )

    def _run() -> SegmentGenerationResult:
        return _grid_generator(
            duration_vocabulary,
            accent_field_sampler=accent_field_sampler,
            hand_coupling_sampler=HandCouplingSampler(
                config=HandCouplingConfig(
                    co_activity_strength=0.6, activity_right=0.7, activity_left=0.7, sync_strength=0.0
                )
            ),
            register_curve_sampler=register_curve_sampler,
            base_duration=Fraction(1, 8),
            figure=_figure([0, 1]),
        ).generate(
            bar_count=3,
            time_numerator=4,
            time_denominator=4,
            grid_count_per_bar=4,
            chord_resolution=1,
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=3),
            rng=default_rng(99),
            source_file=Path("synthetic.mxl"),
        )

    first = _run()
    second = _run()

    assert first.segment.tokens == second.segment.tokens
    assert first.trace == second.trace
