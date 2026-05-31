from collections import Counter
from fractions import Fraction
from pathlib import Path

from numpy.random import default_rng

from musak_model.generation.constraints import GenerationConstraints
from musak_model.harmony.expansion import chord_pitch_class_set
from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.base_durations import BaseDurationDistribution
from musak_model.synthetic.figures import FigureVocabulary
from musak_model.synthetic.processes.accent import AccentFieldConfig, AccentFieldSampler
from musak_model.synthetic.processes.chord_track import ChordTrackSampler, uniform_transition_model
from musak_model.synthetic.processes.hand_coupling import HandCouplingConfig, HandCouplingSampler
from musak_model.synthetic.processes.pitch import RegisterCurveConfig, RegisterCurveSampler
from musak_model.synthetic.substitution import (
    ALL_MELODIC_TEXTURE,
    AccompanimentConfig,
    AccompanimentRhythm,
    HandTexture,
    HandTextureConfig,
    SegmentGenerator,
    SubstitutionConfig,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import degree_pitch_class
from musak_model.tokens.schema import Hand, HandToken, HoldToken, JoinWithPreviousToken, NoteToken, ScaleType, Token

_C_MAJOR = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)


def _figure(positions: list[int]) -> FigureNGram:
    return FigureNGram(onsets=tuple((((position, 0),), Fraction(1)) for position in positions))


def _vocabulary() -> FigureVocabulary:
    figure = _figure([0, 2])
    return FigureVocabulary.from_counts(
        {ScaleType.MAJOR: {Hand.RIGHT: {2: Counter({figure: 1})}, Hand.LEFT: {2: Counter({figure: 1})}}}
    )


def _base_durations() -> BaseDurationDistribution:
    return BaseDurationDistribution(
        weights_by_group={
            (ScaleType.MAJOR, Hand.RIGHT, 2): ((Fraction(1, 2), 1),),
            (ScaleType.MAJOR, Hand.LEFT, 2): ((Fraction(1, 2), 1),),
        }
    )


def _generator(duration_vocabulary: DurationVocabulary, *, texture: HandTextureConfig) -> SegmentGenerator:
    return SegmentGenerator(
        substitution_config=SubstitutionConfig(
            lambda_curve=0.0,
            lambda_harmonic=0.0,
            lambda_accent=0.0,
            lambda_chord_figure=0.0,
            commonness_bias=1.0,
            max_resample_retries=4,
            monophonic=False,
            texture=texture,
        ),
        register_curve_sampler=RegisterCurveSampler(
            config=RegisterCurveConfig(
                arch_basis_count=3, arch_amplitude=0.0, arch_decay=1.0, ou_theta=0.5, ou_sigma=0.0
            )
        ),
        accent_field_sampler=AccentFieldSampler(
            config=AccentFieldConfig(
                baseline_logit=40.0,
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
        chord_track_sampler=ChordTrackSampler(model=uniform_transition_model((_C_MAJOR,))),
        chord_vocabulary=ChordVocabularyConfig.load(),
        figure_vocabulary=_vocabulary(),
        base_duration_distribution=_base_durations(),
        duration_vocabulary=duration_vocabulary,
        figure_lengths=(2,),
    )


def _texture(
    left: HandTexture, *, rhythm: AccompanimentRhythm = AccompanimentRhythm.BLOCK_PER_WINDOW
) -> HandTextureConfig:
    return HandTextureConfig(
        right=HandTexture.MELODIC,
        left=left,
        accompaniment=AccompanimentConfig(rhythm=rhythm, max_chord_notes=3),
    )


def _generate(generator: SegmentGenerator, *, time_numerator: int = 4, grid_count_per_bar: int = 1) -> list[Token]:
    constraints = GenerationConstraints(time_numerator=time_numerator, time_denominator=4, bar_count=1)
    result = generator.generate(
        bar_count=1,
        time_numerator=time_numerator,
        time_denominator=4,
        grid_count_per_bar=grid_count_per_bar,
        chord_resolution=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        constraints=constraints,
        rng=default_rng(7),
        source_file=Path("synthetic.mxl"),
    )
    return result.segment.tokens


def _tokens_under_hand(tokens: list[Token], hand: Hand) -> list[Token]:
    collected: list[Token] = []
    current_hand: Hand | None = None
    for token in tokens:
        if isinstance(token, HandToken):
            current_hand = token.hand
        elif isinstance(token, NoteToken | JoinWithPreviousToken | HoldToken):
            if current_hand == hand:
                collected.append(token)

    return collected


def test_block_chord_hand_locks_every_note_to_the_chord_track(duration_vocabulary: DurationVocabulary) -> None:
    tokens = _generate(_generator(duration_vocabulary, texture=_texture(HandTexture.BLOCK_CHORD)))
    left_notes = [token for token in _tokens_under_hand(tokens, Hand.LEFT) if isinstance(token, NoteToken)]

    chord_pitch_classes = chord_pitch_class_set(
        _C_MAJOR, scale_type=ScaleType.MAJOR, vocabulary=ChordVocabularyConfig.load()
    )
    assert left_notes
    assert all(
        degree_pitch_class(note.degree, note.accidental, scale_type=ScaleType.MAJOR) in chord_pitch_classes
        for note in left_notes
    )
    # a block chord is multiple notes sharing one attack
    assert any(isinstance(token, JoinWithPreviousToken) for token in _tokens_under_hand(tokens, Hand.LEFT))


def test_sustained_bass_hand_emits_only_the_root(duration_vocabulary: DurationVocabulary) -> None:
    tokens = _generate(_generator(duration_vocabulary, texture=_texture(HandTexture.SUSTAINED_BASS)))
    left = _tokens_under_hand(tokens, Hand.LEFT)
    left_notes = [token for token in left if isinstance(token, NoteToken)]

    assert len(left_notes) == 1  # one held root per (single, whole-bar) chord window
    assert degree_pitch_class(left_notes[0].degree, left_notes[0].accidental, scale_type=ScaleType.MAJOR) == 0
    assert not any(isinstance(token, JoinWithPreviousToken) for token in left)


def test_block_chord_window_longer_than_a_single_note_is_held_with_hold_tokens(
    duration_vocabulary: DurationVocabulary,
) -> None:
    # 5/4 bar at whole-note chord resolution → a 5/4 window = whole + quarter, so the chord is held by a HoldToken.
    tokens = _generate(
        _generator(duration_vocabulary, texture=_texture(HandTexture.BLOCK_CHORD)),
        time_numerator=5,
        grid_count_per_bar=5,
    )
    left = _tokens_under_hand(tokens, Hand.LEFT)

    assert any(isinstance(token, HoldToken) for token in left)


def test_melodic_hand_is_unchanged_when_the_other_hand_accompanies(duration_vocabulary: DurationVocabulary) -> None:
    accompanied = _generate(_generator(duration_vocabulary, texture=_texture(HandTexture.BLOCK_CHORD)))
    all_melodic = _generate(_generator(duration_vocabulary, texture=ALL_MELODIC_TEXTURE))

    assert _tokens_under_hand(accompanied, Hand.RIGHT) == _tokens_under_hand(all_melodic, Hand.RIGHT)


def test_substitution_config_defaults_to_all_melodic() -> None:
    config = SubstitutionConfig(
        lambda_curve=0.0,
        lambda_harmonic=0.0,
        lambda_accent=0.0,
        lambda_chord_figure=0.0,
        commonness_bias=1.0,
        max_resample_retries=4,
        monophonic=False,
    )

    assert config.texture == ALL_MELODIC_TEXTURE
    assert config.texture.texture(Hand.RIGHT) is HandTexture.MELODIC
    assert config.texture.texture(Hand.LEFT) is HandTexture.MELODIC
