from bisect import bisect_right
from collections.abc import Callable, Mapping
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
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.base_durations import (
    BaseDurationDistribution,
    BaseDurationWeight,
    weighted_base_duration_choice,
)
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry
from musak_model.synthetic.harmony.expansion import chord_pitch_class_set
from musak_model.synthetic.harmony.schema import Chord
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.harmony.windows import chord_window_grid
from musak_model.synthetic.processes.accent import AccentFieldSampler
from musak_model.synthetic.processes.chord_track import ChordTrackSampler
from musak_model.synthetic.processes.hand_coupling import HandCouplingSampler
from musak_model.synthetic.processes.pitch import RegisterCurveSampler
from musak_model.synthetic.substitution.config import SubstitutionConfig
from musak_model.synthetic.substitution.emission import anchor_figure_to_tokens
from musak_model.synthetic.substitution.sampling import sample_substituted_figure
from musak_model.synthetic.substitution.trace import BaselineSample, GenerationTrace, SegmentGenerationResult
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import diatonic_position_to_degree_and_octave, note_token_to_midi_pitch
from musak_model.tokens.schema import (
    MIN_DURATION_ID,
    BarToken,
    EndToken,
    Hand,
    HandToken,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
    scale_size_for_type,
)
from musak_shared.misc import is_power_of_two

type FigureEntriesByGroup = Mapping[tuple[Hand, int], tuple[FigureVocabularyEntry, ...]]
type ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class SegmentGenerator:
    substitution_config: SubstitutionConfig
    register_curve_sampler: RegisterCurveSampler
    accent_field_sampler: AccentFieldSampler
    hand_coupling_sampler: HandCouplingSampler
    chord_track_sampler: ChordTrackSampler
    chord_vocabulary: ChordVocabularyConfig
    figure_vocabulary: FigureVocabulary
    base_duration_distribution: BaseDurationDistribution
    duration_vocabulary: DurationVocabulary
    figure_lengths: tuple[int, ...]

    def generate(
        self,
        *,
        bar_count: int,
        time_numerator: int,
        time_denominator: int,
        grid_count_per_bar: int,
        chord_resolution: int,
        scale_root: int,
        scale_type: ScaleType,
        constraints: GenerationConstraints,
        rng: Generator,
        source_file: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> SegmentGenerationResult:
        if bar_count <= 0:
            raise ValueError("bar_count must be positive")

        if grid_count_per_bar <= 0:
            raise ValueError("grid_count_per_bar must be positive")

        if not is_power_of_two(chord_resolution):
            raise ValueError("chord_resolution must be a power of two note value (1 whole, 2 half, 4 quarter, ...)")

        if not self.figure_lengths:
            raise ValueError("figure_lengths must be non-empty")

        bar_duration = Fraction(time_numerator, time_denominator)
        cell_duration = bar_duration / grid_count_per_bar
        cell_count = bar_count * grid_count_per_bar
        entries_by_group = self._figure_entries_by_group(scale_type)
        right_curve = self.register_curve_sampler.sample(
            length=cell_count,
            scale_type=scale_type,
            hand=Hand.RIGHT,
            rng=rng,
        )
        left_curve = self.register_curve_sampler.sample(
            length=cell_count,
            scale_type=scale_type,
            hand=Hand.LEFT,
            rng=rng,
        )
        right_weights = self.accent_field_sampler.sample_weights(
            bar_count=bar_count,
            grid_count_per_bar=grid_count_per_bar,
            rng=rng,
        )
        left_weights = self.accent_field_sampler.sample_weights(
            bar_count=bar_count,
            grid_count_per_bar=grid_count_per_bar,
            rng=rng,
        )
        gates = self.hand_coupling_sampler.sample_gates(
            cell_count=cell_count,
            rng=rng,
        )
        onsets = self.hand_coupling_sampler.sample_onsets(
            right_weights=right_weights,
            left_weights=left_weights,
            rng=rng,
        )
        chord_windows = chord_window_grid(
            measure_duration=bar_duration,
            total_duration=bar_duration * bar_count,
            resolution=chord_resolution,
        )
        chord_track = self.chord_track_sampler.sample(
            length=len(chord_windows),
            rng=rng,
        )
        cell_chord_pitch_classes = self._cell_chord_pitch_classes(
            chord_windows=chord_windows,
            chord_track=chord_track,
            scale_type=scale_type,
            cell_duration=cell_duration,
            cell_count=cell_count,
        )

        scale_size = scale_size_for_type(scale_type)
        baseline_samples: list[BaselineSample] = []
        for cell_index in range(cell_count):
            bar_index, position = divmod(cell_index, grid_count_per_bar)
            for hand, curve, weights in (
                (Hand.RIGHT, right_curve, right_weights),
                (Hand.LEFT, left_curve, left_weights),
            ):
                anchor = int(curve[cell_index])
                baseline_samples.append(
                    BaselineSample(
                        hand=hand,
                        bar_index=bar_index,
                        position=position,
                        start_in_bars=1 + bar_index + position / grid_count_per_bar,
                        register_anchor=anchor,
                        register_midi_pitch=self._register_midi_pitch(
                            anchor=anchor,
                            scale_size=scale_size,
                            scale_root=scale_root,
                            scale_type=scale_type,
                            hand=hand,
                        ),
                        accent_weight=weights[cell_index],
                    )
                )

        state = GenerationConstraintState(constraints=constraints)
        tokens: list[Token] = []
        for bar_index in range(bar_count):
            for hand, curve, weights in (
                (Hand.RIGHT, right_curve, right_weights),
                (Hand.LEFT, left_curve, left_weights),
            ):
                state, tokens = self._append(state, tokens, HandToken(hand=hand))
                state, tokens = self._emit_hand_bar(
                    state=state,
                    tokens=tokens,
                    hand=hand,
                    bar_index=bar_index,
                    grid_count_per_bar=grid_count_per_bar,
                    cell_duration=cell_duration,
                    curve=curve,
                    weights=weights,
                    onsets=onsets,
                    gates=gates,
                    entries_by_group=entries_by_group,
                    scale_type=scale_type,
                    cell_chord_pitch_classes=cell_chord_pitch_classes,
                    rng=rng,
                )
            state, tokens = self._append(state, tokens, BarToken())
            if progress_callback is not None:
                progress_callback(bar_index + 1, bar_count)

        state, tokens = self._append(state, tokens, EndToken())
        segment = Segment(
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
        trace = GenerationTrace(
            samples=tuple(baseline_samples),
            grid_count_per_bar=grid_count_per_bar,
            bar_count=bar_count,
        )
        return SegmentGenerationResult(segment=segment, trace=trace)

    @staticmethod
    def _register_midi_pitch(
        *,
        anchor: int,
        scale_size: int,
        scale_root: int,
        scale_type: ScaleType,
        hand: Hand,
    ) -> int:
        degree, octave_offset = diatonic_position_to_degree_and_octave(anchor, scale_size=scale_size)
        note_token = NoteToken(
            degree=degree,
            accidental=0,
            octave_offset=octave_offset,
            duration_id=MIN_DURATION_ID,
        )
        return note_token_to_midi_pitch(note_token, scale_root=scale_root, scale_type=scale_type, hand=hand)

    def _cell_chord_pitch_classes(
        self,
        *,
        chord_windows: tuple[tuple[Fraction, Fraction], ...],
        chord_track: tuple[Chord, ...],
        scale_type: ScaleType,
        cell_duration: Fraction,
        cell_count: int,
    ) -> tuple[frozenset[int], ...]:
        window_starts = [start for start, _ in chord_windows]
        pitch_classes_per_window = tuple(
            chord_pitch_class_set(chord, scale_type=scale_type, vocabulary=self.chord_vocabulary)
            for chord in chord_track
        )
        return tuple(
            pitch_classes_per_window[bisect_right(window_starts, cell_index * cell_duration) - 1]
            for cell_index in range(cell_count)
        )

    def _figure_entries_by_group(self, scale_type: ScaleType) -> FigureEntriesByGroup:
        figure_lengths = frozenset(self.figure_lengths)
        grouped: dict[tuple[Hand, int], list[FigureVocabularyEntry]] = {}
        for entry in self.figure_vocabulary.entries:
            group = entry.group
            if group.scale_type != scale_type or group.n not in figure_lengths:
                continue

            grouped.setdefault((group.hand, group.n), []).append(entry)

        return {key: tuple(entries) for key, entries in grouped.items()}

    def _emit_hand_bar(
        self,
        *,
        state: GenerationConstraintState,
        tokens: list[Token],
        hand: Hand,
        bar_index: int,
        grid_count_per_bar: int,
        cell_duration: Fraction,
        curve: tuple[int, ...],
        weights: tuple[float, ...],
        onsets: tuple[Mapping[Hand, bool], ...],
        gates: tuple[Mapping[Hand, bool], ...],
        entries_by_group: FigureEntriesByGroup,
        scale_type: ScaleType,
        cell_chord_pitch_classes: tuple[frozenset[int], ...],
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        """Fill an active hand's bar from a sub-bar onset grid.

        A cell fires iff its onset mask and the hand-coupling gate are both active for the hand. The
        cursor walks the bar: stretches before the next fired cell become rests, a fired cell starts a
        figure whose register anchor, slope target, and accent value are read at that cell. A figure may
        span many cells; the cursor advances to its end and onsets it masks are skipped.

        NEEDS IMPROVEMENT: figures are still placed greedily with no lookahead. When no sampled figure fits
        the remaining bar time the trailing gap becomes a rest.
        """
        bar_start = state.constraints.bar_start(bar_index)
        bar_end = state.constraints.bar_end(bar_index)
        while True:
            cursor = state.cursor(hand)
            if cursor >= bar_end:
                break

            fired_cell_index = self._next_fired_cell(
                hand=hand,
                bar_index=bar_index,
                grid_count_per_bar=grid_count_per_bar,
                cell_duration=cell_duration,
                bar_start=bar_start,
                cursor=cursor,
                onsets=onsets,
                gates=gates,
            )
            if fired_cell_index is None:
                return self._fill_with_rest(state=state, tokens=tokens, hand=hand, duration=bar_end - cursor)

            onset_time = bar_start + (fired_cell_index % grid_count_per_bar) * cell_duration
            if onset_time > cursor:
                state, tokens = self._fill_with_rest(
                    state=state, tokens=tokens, hand=hand, duration=onset_time - cursor
                )

            placement = self._place_one_figure(
                state=state,
                hand=hand,
                entries_by_group=entries_by_group,
                scale_type=scale_type,
                chord_pitch_classes=cell_chord_pitch_classes[fired_cell_index],
                curve=curve,
                fired_cell_index=fired_cell_index,
                envelope_value=weights[fired_cell_index],
                remaining=bar_end - state.cursor(hand),
                rng=rng,
            )
            if placement is None:
                return self._fill_with_rest(
                    state=state, tokens=tokens, hand=hand, duration=bar_end - state.cursor(hand)
                )

            state, placed_tokens = placement
            tokens = tokens + placed_tokens

        return state, tokens

    @staticmethod
    def _next_fired_cell(
        *,
        hand: Hand,
        bar_index: int,
        grid_count_per_bar: int,
        cell_duration: Fraction,
        bar_start: Fraction,
        cursor: Fraction,
        onsets: tuple[Mapping[Hand, bool], ...],
        gates: tuple[Mapping[Hand, bool], ...],
    ) -> int | None:
        current_position = max(0, (cursor - bar_start) // cell_duration)
        for position in range(int(current_position), grid_count_per_bar):
            onset_time = bar_start + position * cell_duration
            if onset_time < cursor:
                continue

            cell_index = bar_index * grid_count_per_bar + position
            if onsets[cell_index][hand] and gates[cell_index][hand]:
                return cell_index

        return None

    def _place_one_figure(
        self,
        *,
        state: GenerationConstraintState,
        hand: Hand,
        entries_by_group: FigureEntriesByGroup,
        scale_type: ScaleType,
        chord_pitch_classes: frozenset[int],
        curve: tuple[int, ...],
        fired_cell_index: int,
        envelope_value: float,
        remaining: Fraction,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token]] | None:
        anchor = int(curve[fired_cell_index])
        for _ in range(self.substitution_config.max_resample_retries):
            figure_length = int(rng.choice(self.figure_lengths))
            span_cell_index = min(fired_cell_index + figure_length - 1, len(curve) - 1)
            target_slope = int(curve[span_cell_index]) - anchor
            entries = entries_by_group.get((hand, figure_length), ())
            candidate_bases = self.base_duration_distribution.candidates(
                scale_type=scale_type, hand=hand, figure_length=figure_length
            )
            if not entries or not candidate_bases:
                continue

            entry = sample_substituted_figure(
                entries=entries,
                anchor=anchor,
                target_slope=target_slope,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
                envelope_value=envelope_value,
                config=self.substitution_config,
                rng=rng,
            )
            fitting_bases = self._fitting_base_durations(entry.figure, candidate_bases, remaining=remaining)
            if not fitting_bases:
                continue

            base_duration = weighted_base_duration_choice(fitting_bases, rng=rng)
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
                    trial_state = trial_state.apply(token, duration_vocabulary=self.duration_vocabulary)
            except GenerationConstraintError:
                continue

            return trial_state, candidate_tokens

        return None

    def _fitting_base_durations(
        self,
        figure: FigureNGram,
        candidate_bases: tuple[BaseDurationWeight, ...],
        *,
        remaining: Fraction,
    ) -> list[BaseDurationWeight]:
        normalized_durations = frozenset(duration for _, duration in figure.onsets)
        span_units = sum((duration for _, duration in figure.onsets), Fraction(0))
        fitting: list[BaseDurationWeight] = []
        for base_duration, count in candidate_bases:
            if span_units * base_duration > remaining:
                continue

            if all(
                self.duration_vocabulary.duration_id_or_none(normalized * base_duration) is not None
                for normalized in normalized_durations
            ):
                fitting.append((base_duration, count))

        return fitting

    def _fill_with_rest(
        self,
        *,
        state: GenerationConstraintState,
        tokens: list[Token],
        hand: Hand,
        duration: Fraction,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        rest_tokens = self._rest_tokens_for_duration(duration)
        if rest_tokens is None:
            raise GenerationConstraintError(
                f"could not fill {duration} of the {hand.value} hand bar with vocabulary rests"
            )

        for token in rest_tokens:
            state, tokens = self._append(state, tokens, token)

        return state, tokens

    def _rest_tokens_for_duration(self, duration: Fraction) -> list[Token] | None:
        direct_id = self.duration_vocabulary.duration_id_or_none(duration)
        if direct_id is not None:
            return [RestToken(duration_id=direct_id)]

        rest_tokens: list[Token] = []
        remaining = duration
        for fraction in sorted(self.duration_vocabulary.all_fractions(), reverse=True):
            while fraction <= remaining:
                rest_tokens.append(RestToken(duration_id=self.duration_vocabulary.require_duration_id(fraction)))
                remaining -= fraction

        if remaining != 0:
            return None

        return rest_tokens

    def _append(
        self,
        state: GenerationConstraintState,
        tokens: list[Token],
        token: Token,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        new_state = state.apply(token, duration_vocabulary=self.duration_vocabulary)
        return new_state, tokens + [token]
