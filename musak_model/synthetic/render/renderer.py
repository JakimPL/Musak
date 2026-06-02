from collections.abc import Sequence
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
from musak_model.harmony.expansion import chord_pitch_class_set
from musak_model.harmony.schema import Chord
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.base_durations import BaseDurationDistribution, choose_base_duration
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry
from musak_model.synthetic.processes.density import RhythmicDensitySampler
from musak_model.synthetic.processes.pitch import RegisterCurveSampler
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.figure_selection import figure_span_units, select_figure
from musak_model.synthetic.render.motif import MotifConfig
from musak_model.synthetic.render.slots import RenderSlot, render_slots
from musak_model.synthetic.structure.form import FormTree
from musak_model.synthetic.structure.harmony_grammar import HarmonyGrammarSampler
from musak_model.synthetic.structure.meter import (
    MetricalLeafType,
    MetricalNode,
    MetricalTree,
    MetricalTreeSampler,
)
from musak_model.synthetic.substitution.emission import anchor_figure_to_tokens, rest_tokens
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, ScaleType, Token

_HANDS: tuple[Hand, ...] = (Hand.RIGHT, Hand.LEFT)


def phrase_harmony(
    frontier: Sequence[MetricalNode],
    form: FormTree,
    *,
    harmony_sampler: HarmonyGrammarSampler,
    scale_type: ScaleType,
    bar_duration: Fraction,
    rng: Generator,
) -> tuple[Chord, ...]:
    chords: list[Chord] = []
    for phrase in form.phrases:
        phrase_start = phrase.start_bar * bar_duration
        phrase_end = (phrase.start_bar + phrase.bar_span) * bar_duration
        slot_count = sum(1 for node in frontier if phrase_start <= node.offset < phrase_end)
        harmony = harmony_sampler.sample(
            slot_count=slot_count,
            scale_type=scale_type,
            closing=phrase.closing,
            rng=rng,
        )
        chords.extend(harmony.chords())

    return tuple(chords)


def class_metrical_tree(
    metrical_sampler: MetricalTreeSampler,
    form: FormTree,
    *,
    time_numerator: int,
    time_denominator: int,
    rng: Generator,
) -> MetricalTree:
    bar_duration = Fraction(time_numerator, time_denominator)
    class_trees: dict[tuple[int, int], MetricalTree] = {}
    bars: list[MetricalNode] = []
    for segment in form.segments:
        key = (segment.class_label, segment.bar_span)
        class_tree = class_trees.get(key)
        if class_tree is None:
            class_tree = metrical_sampler.sample(
                time_numerator=time_numerator,
                time_denominator=time_denominator,
                bar_count=segment.bar_span,
                rng=rng,
            )
            class_trees[key] = class_tree

        offset = segment.start_bar * bar_duration
        bars.extend(_translate_node(bar, offset) for bar in class_tree.bars)

    return MetricalTree(time_numerator, time_denominator, tuple(bars))


def _translate_node(node: MetricalNode, offset: Fraction) -> MetricalNode:
    return MetricalNode(
        offset=node.offset + offset,
        duration=node.duration,
        weight=node.weight,
        children=tuple(_translate_node(child, offset) for child in node.children),
        leaf_type=node.leaf_type,
    )


def _target_slope(anchors: tuple[int, ...], index: int) -> int:
    return anchors[index + 1] - anchors[index] if index + 1 < len(anchors) else 0


@dataclass(frozen=True)
class SurfaceRenderer:
    config: RenderConfig
    metrical_sampler: MetricalTreeSampler
    harmony_sampler: HarmonyGrammarSampler
    register_curve_sampler: RegisterCurveSampler
    figure_vocabulary: FigureVocabulary
    duration_vocabulary: DurationVocabulary
    chord_vocabulary: ChordVocabularyConfig
    base_duration_distribution: BaseDurationDistribution
    rhythmic_density_sampler: RhythmicDensitySampler
    motif_config: MotifConfig

    def render(
        self,
        *,
        time_numerator: int,
        time_denominator: int,
        scale_root: int,
        scale_type: ScaleType,
        form: FormTree,
        harmonic_slot_duration: Fraction,
        constraints: GenerationConstraints,
        source_file: Path,
        rng: Generator,
    ) -> Segment:
        bar_count = form.bar_count
        bar_duration = Fraction(time_numerator, time_denominator)
        tree = class_metrical_tree(
            self.metrical_sampler,
            form,
            time_numerator=time_numerator,
            time_denominator=time_denominator,
            rng=rng,
        )
        frontier = tree.harmonic_frontier(harmonic_slot_duration)
        chords = phrase_harmony(
            frontier,
            form,
            harmony_sampler=self.harmony_sampler,
            scale_type=scale_type,
            bar_duration=bar_duration,
            rng=rng,
        )
        slots = render_slots(tree, chords, slot_duration=harmonic_slot_duration)

        shortest_note_duration = min(self.duration_vocabulary.all_fractions())
        chord_pitch_classes = tuple(
            chord_pitch_class_set(slot.chord, scale_type=scale_type, vocabulary=self.chord_vocabulary) for slot in slots
        )
        anchors = {
            hand: self.register_curve_sampler.sample(length=len(slots), scale_type=scale_type, hand=hand, rng=rng)
            for hand in _HANDS
        }
        density = {hand: self.rhythmic_density_sampler.sample(length=len(slots), rng=rng) for hand in _HANDS}
        hand_entries = {
            hand: tuple(self.figure_vocabulary.filter(scale_type=scale_type, hand=hand).entries) for hand in _HANDS
        }

        state = GenerationConstraintState(constraints=constraints)
        tokens: list[Token] = []
        for bar_index in range(bar_count):
            bar_slot_indices = [
                index
                for index, slot in enumerate(slots)
                if bar_index * bar_duration <= slot.offset < (bar_index + 1) * bar_duration
            ]
            for hand in _HANDS:
                state, tokens = self._append(state, tokens, HandToken(hand=hand))
                for slot_index in bar_slot_indices:
                    state, tokens = self._render_slot(
                        state=state,
                        tokens=tokens,
                        slots=slots,
                        slot_index=slot_index,
                        hand=hand,
                        anchors=anchors[hand],
                        density_offset=density[hand][slot_index],
                        entries=hand_entries[hand],
                        chord_pitch_classes=chord_pitch_classes[slot_index],
                        scale_type=scale_type,
                        shortest_note_duration=shortest_note_duration,
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

    def _render_slot(
        self,
        *,
        state: GenerationConstraintState,
        tokens: list[Token],
        slots: tuple[RenderSlot, ...],
        slot_index: int,
        hand: Hand,
        anchors: tuple[int, ...],
        density_offset: float,
        entries: tuple[FigureVocabularyEntry, ...],
        chord_pitch_classes: frozenset[int],
        scale_type: ScaleType,
        shortest_note_duration: Fraction,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        slot = slots[slot_index]
        if slot.leaf_type is MetricalLeafType.REST:
            return self._emit_or_rest(state, tokens, rest_tokens(slot.duration, self.duration_vocabulary), slot)

        return self._fill_sound_slot(
            state=state,
            tokens=tokens,
            slot=slot,
            hand=hand,
            anchor=anchors[slot_index],
            target_slope=_target_slope(anchors, slot_index),
            density_offset=density_offset,
            entries=entries,
            chord_pitch_classes=chord_pitch_classes,
            scale_type=scale_type,
            shortest_note_duration=shortest_note_duration,
            rng=rng,
        )

    def _fill_sound_slot(
        self,
        *,
        state: GenerationConstraintState,
        tokens: list[Token],
        slot: RenderSlot,
        hand: Hand,
        anchor: int,
        target_slope: int,
        density_offset: float,
        entries: tuple[FigureVocabularyEntry, ...],
        chord_pitch_classes: frozenset[int],
        scale_type: ScaleType,
        shortest_note_duration: Fraction,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        remaining = slot.duration
        while remaining >= shortest_note_duration:
            feasible = tuple(
                entry
                for entry in entries
                if self._fitting_base(
                    entry, hand=hand, scale_type=scale_type, density_offset=density_offset, remaining=remaining
                )
                is not None
            )
            if not feasible:
                break

            state, tokens, consumed = self._place_one_figure(
                state=state,
                tokens=tokens,
                feasible=feasible,
                hand=hand,
                anchor=anchor,
                target_slope=target_slope,
                density_offset=density_offset,
                remaining=remaining,
                weight=slot.weight,
                chord_pitch_classes=chord_pitch_classes,
                scale_type=scale_type,
                rng=rng,
            )
            if consumed is None:
                break

            remaining -= consumed

        return self._pad_with_rest(state, tokens, remaining)

    def _place_one_figure(
        self,
        *,
        state: GenerationConstraintState,
        tokens: list[Token],
        feasible: tuple[FigureVocabularyEntry, ...],
        hand: Hand,
        anchor: int,
        target_slope: int,
        density_offset: float,
        remaining: Fraction,
        weight: float,
        chord_pitch_classes: frozenset[int],
        scale_type: ScaleType,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token], Fraction | None]:
        for _ in range(self.config.max_resample_retries):
            entry = select_figure(
                feasible,
                anchor=anchor,
                target_slope=target_slope,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
                weight=weight,
                config=self.config,
                rng=rng,
            )
            base_duration = self._fitting_base(
                entry, hand=hand, scale_type=scale_type, density_offset=density_offset, remaining=remaining
            )
            if base_duration is None:
                continue

            candidate = anchor_figure_to_tokens(
                figure=entry.figure,
                anchor=anchor,
                base_duration=base_duration,
                scale_type=scale_type,
                duration_vocabulary=self.duration_vocabulary,
            )
            applied = self._try_apply(state, candidate)
            if applied is not None:
                return applied, tokens + candidate, figure_span_units(entry.figure) * base_duration

        return state, tokens, None

    def _fitting_base(
        self,
        entry: FigureVocabularyEntry,
        *,
        hand: Hand,
        scale_type: ScaleType,
        density_offset: float,
        remaining: Fraction,
    ) -> Fraction | None:
        return choose_base_duration(
            entry.figure,
            self.base_duration_distribution.candidates(scale_type=scale_type, hand=hand, figure_length=entry.figure.n),
            density_offset=density_offset,
            remaining=remaining,
            duration_vocabulary=self.duration_vocabulary,
        )

    def _pad_with_rest(
        self,
        state: GenerationConstraintState,
        tokens: list[Token],
        remaining: Fraction,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        if remaining <= 0:
            return state, tokens

        rest = rest_tokens(remaining, self.duration_vocabulary)
        applied = self._try_apply(state, rest) if rest is not None else None
        if applied is None:
            raise GenerationConstraintError(f"could not fill a {remaining} remainder with vocabulary durations")

        return applied, tokens + (rest or [])

    def _emit_or_rest(
        self,
        state: GenerationConstraintState,
        tokens: list[Token],
        candidate: list[Token] | None,
        slot: RenderSlot,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        if candidate is not None:
            applied = self._try_apply(state, candidate)
            if applied is not None:
                return applied, tokens + candidate

        rest = rest_tokens(slot.duration, self.duration_vocabulary)
        applied = self._try_apply(state, rest) if rest is not None else None
        if applied is None:
            raise GenerationConstraintError(f"could not fill a {slot.duration} slot with vocabulary durations")

        return applied, tokens + (rest or [])

    def _try_apply(self, state: GenerationConstraintState, candidate: list[Token]) -> GenerationConstraintState | None:
        trial_state = state
        try:
            for token in candidate:
                trial_state = trial_state.apply(token, duration_vocabulary=self.duration_vocabulary)
        except GenerationConstraintError:
            return None

        return trial_state

    def _append(
        self, state: GenerationConstraintState, tokens: list[Token], token: Token
    ) -> tuple[GenerationConstraintState, list[Token]]:
        return state.apply(token, duration_vocabulary=self.duration_vocabulary), tokens + [token]
