from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from numpy.random import Generator

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.generation.constraints import (
    GenerationConstraintError,
    GenerationConstraints,
    GenerationConstraintState,
)
from musak_model.synthetic.figures import FigureVocabulary
from musak_model.synthetic.harmony.expansion import chord_pitch_class_set
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.processes.chord_track import ChordTrackSampler
from musak_model.synthetic.processes.pitch import RegisterCurveSampler
from musak_model.synthetic.substitution.config import SubstitutionConfig
from musak_model.synthetic.substitution.emission import anchor_figure_to_tokens
from musak_model.synthetic.substitution.sampling import (
    monorhythmic_entries,
    sample_substituted_figure,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, ScaleType, Token


@dataclass(frozen=True)
class SegmentGenerator:
    substitution_config: SubstitutionConfig
    register_curve_sampler: RegisterCurveSampler
    chord_track_sampler: ChordTrackSampler
    chord_vocabulary: ChordVocabularyConfig
    figure_vocabulary: FigureVocabulary
    duration_vocabulary: DurationVocabulary
    figure_lengths: tuple[int, ...]

    def generate(
        self,
        *,
        bar_count: int,
        time_numerator: int,
        time_denominator: int,
        scale_root: int,
        scale_type: ScaleType,
        constraints: GenerationConstraints,
        rng: Generator,
        source_file: Path,
    ) -> Segment:
        if bar_count <= 0:
            raise ValueError("bar_count must be positive")

        if not self.figure_lengths:
            raise ValueError("figure_lengths must be non-empty")

        bar_duration = Fraction(time_numerator, time_denominator)
        right_curve = self.register_curve_sampler.sample(
            length=bar_count,
            scale_type=scale_type,
            hand=Hand.RIGHT,
            rng=rng,
        )
        left_curve = self.register_curve_sampler.sample(
            length=bar_count,
            scale_type=scale_type,
            hand=Hand.LEFT,
            rng=rng,
        )
        chord_track = self.chord_track_sampler.sample(
            length=bar_count,
            rng=rng,
        )

        state = GenerationConstraintState(constraints=constraints)
        tokens: list[Token] = []
        for bar_index in range(bar_count):
            chord_pcs = chord_pitch_class_set(
                chord_track[bar_index],
                scale_type=scale_type,
                vocabulary=self.chord_vocabulary,
            )
            for hand, curve in ((Hand.RIGHT, right_curve), (Hand.LEFT, left_curve)):
                anchor = int(curve[bar_index])
                next_anchor = int(curve[bar_index + 1]) if bar_index + 1 < bar_count else anchor
                state, tokens = self._append(state, tokens, HandToken(hand=hand))
                state, tokens = self._emit_hand_bar(
                    state=state,
                    tokens=tokens,
                    hand=hand,
                    bar_duration=bar_duration,
                    scale_type=scale_type,
                    chord_pitch_classes=chord_pcs,
                    anchor=anchor,
                    target_slope=next_anchor - anchor,
                    rng=rng,
                )
            state, tokens = self._append(state, tokens, BarToken())

        state, tokens = self._append(state, tokens, EndToken())
        return Segment(
            tokens=tokens,
            metadata=SegmentMetadata(
                scale_root=scale_root,
                scale_type=scale_type,
                time_numerator=time_numerator,
                time_denominator=time_denominator,
                bar_count=bar_count,
                window_start_bar=0,
                source_file=source_file,
                difficulty_level=None,
            ),
        )

    def _emit_hand_bar(
        self,
        *,
        state: GenerationConstraintState,
        tokens: list[Token],
        hand: Hand,
        bar_duration: Fraction,
        scale_type: ScaleType,
        chord_pitch_classes: frozenset[int],
        anchor: int,
        target_slope: int,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        for _ in range(self.substitution_config.max_resample_retries):
            figure_length = int(rng.choice(self.figure_lengths))
            base_duration = bar_duration / figure_length
            if self.duration_vocabulary.duration_id_or_none(base_duration) is None:
                continue

            entries = monorhythmic_entries(
                self.figure_vocabulary,
                scale_type=scale_type,
                hand=hand,
                figure_length=figure_length,
            )
            if not entries:
                continue

            entry = sample_substituted_figure(
                entries=entries,
                anchor=anchor,
                target_slope=target_slope,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
                config=self.substitution_config,
                rng=rng,
            )
            candidate_tokens = anchor_figure_to_tokens(
                figure=entry.figure,
                anchor=anchor,
                base_duration=base_duration,
                scale_type=scale_type,
                duration_vocabulary=self.duration_vocabulary,
            )
            trial_state = state
            try:
                for token in candidate_tokens:
                    trial_state = trial_state.apply(
                        token,
                        duration_vocabulary=self.duration_vocabulary,
                    )
            except GenerationConstraintError:
                continue

            return trial_state, tokens + candidate_tokens

        raise GenerationConstraintError(
            f"could not place a figure in the {hand.value} hand within "
            f"{self.substitution_config.max_resample_retries} retries"
        )

    def _append(
        self,
        state: GenerationConstraintState,
        tokens: list[Token],
        token: Token,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        new_state = state.apply(token, duration_vocabulary=self.duration_vocabulary)
        return new_state, tokens + [token]
