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
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry
from musak_model.synthetic.processes.pitch import RegisterCurveSampler
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.figure_selection import figure_fits_slot, select_figure, slot_base_duration
from musak_model.synthetic.render.slots import RenderSlot, render_slots
from musak_model.synthetic.structure.form import FormTree
from musak_model.synthetic.structure.harmony_grammar import HarmonyGrammarSampler
from musak_model.synthetic.structure.meter import (
    MetricalLeafType,
    MetricalNode,
    MetricalTree,
    MetricalTreeSampler,
)
from musak_model.synthetic.substitution.emission import anchor_figure_to_tokens, hold_tokens, rest_tokens
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


@dataclass(frozen=True)
class SurfaceRenderer:
    config: RenderConfig
    metrical_sampler: MetricalTreeSampler
    harmony_sampler: HarmonyGrammarSampler
    register_curve_sampler: RegisterCurveSampler
    figure_vocabulary: FigureVocabulary
    duration_vocabulary: DurationVocabulary
    chord_vocabulary: ChordVocabularyConfig

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
                        anchors=anchors[hand],
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
        anchors: tuple[int, ...],
        entries: tuple[FigureVocabularyEntry, ...],
        chord_pitch_classes: frozenset[int],
        scale_type: ScaleType,
        shortest_note_duration: Fraction,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        slot = slots[slot_index]
        match slot.leaf_type:
            case MetricalLeafType.REST:
                return self._emit_or_rest(state, tokens, rest_tokens(slot.duration, self.duration_vocabulary), slot)
            case MetricalLeafType.TIE:
                return self._emit_or_rest(state, tokens, hold_tokens(slot.duration, self.duration_vocabulary), slot)
            case MetricalLeafType.SOUND:
                return self._render_sound_slot(
                    state=state,
                    tokens=tokens,
                    slots=slots,
                    slot_index=slot_index,
                    anchors=anchors,
                    entries=entries,
                    chord_pitch_classes=chord_pitch_classes,
                    scale_type=scale_type,
                    shortest_note_duration=shortest_note_duration,
                    rng=rng,
                )

    def _render_sound_slot(
        self,
        *,
        state: GenerationConstraintState,
        tokens: list[Token],
        slots: tuple[RenderSlot, ...],
        slot_index: int,
        anchors: tuple[int, ...],
        entries: tuple[FigureVocabularyEntry, ...],
        chord_pitch_classes: frozenset[int],
        scale_type: ScaleType,
        shortest_note_duration: Fraction,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        slot = slots[slot_index]
        anchor = anchors[slot_index]
        target_slope = anchors[slot_index + 1] - anchor if slot_index + 1 < len(anchors) else 0
        feasible = tuple(
            entry
            for entry in entries
            if figure_fits_slot(
                entry.figure,
                slot_duration=slot.duration,
                shortest_note_duration=shortest_note_duration,
                duration_vocabulary=self.duration_vocabulary,
            )
        )
        for _ in range(self.config.max_resample_retries):
            if not feasible:
                break

            entry = select_figure(
                feasible,
                anchor=anchor,
                target_slope=target_slope,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
                weight=slot.weight,
                config=self.config,
                rng=rng,
            )
            candidate = anchor_figure_to_tokens(
                figure=entry.figure,
                anchor=anchor,
                base_duration=slot_base_duration(entry.figure, slot.duration),
                scale_type=scale_type,
                duration_vocabulary=self.duration_vocabulary,
            )
            applied = self._try_apply(state, candidate)
            if applied is not None:
                return applied, tokens + candidate

        return self._emit_or_rest(state, tokens, rest_tokens(slot.duration, self.duration_vocabulary), slot)

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
