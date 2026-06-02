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
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry
from musak_model.synthetic.processes.pitch import RegisterCurveSampler
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.figure_selection import figure_fits_slot, select_figure, slot_base_duration
from musak_model.synthetic.render.motif import MotifConfig, MotifSchema, MotifSlot, ground_motif, select_motif_seed
from musak_model.synthetic.render.slots import RenderSlot, render_slots
from musak_model.synthetic.render.variation import vary_motif
from musak_model.synthetic.structure.form import FormTree, SegmentNode, VariationKind
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


@dataclass(frozen=True)
class _SlotMotif:
    figure: FigureNGram
    reuse: bool


@dataclass(frozen=True)
class _ClassSchema:
    schema: MotifSchema
    sound_slot_count: int


@dataclass(frozen=True)
class _MotifPlanContext:
    slots: tuple[RenderSlot, ...]
    sound_slot_indices: tuple[int, ...]
    chord_pitch_classes: tuple[frozenset[int], ...]
    shortest_note_duration: Fraction
    scale_type: ScaleType
    bar_duration: Fraction


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


def _segment_sound_indices(segment: SegmentNode, context: _MotifPlanContext) -> list[int]:
    start = segment.start_bar * context.bar_duration
    end = (segment.start_bar + segment.bar_span) * context.bar_duration
    return [index for index in context.sound_slot_indices if start <= context.slots[index].offset < end]


def _is_restatement(stored: _ClassSchema, segment: SegmentNode, segment_indices: list[int]) -> bool:
    return stored.sound_slot_count == len(segment_indices) and segment.variation is not VariationKind.FRESH


@dataclass(frozen=True)
class SurfaceRenderer:
    config: RenderConfig
    metrical_sampler: MetricalTreeSampler
    harmony_sampler: HarmonyGrammarSampler
    register_curve_sampler: RegisterCurveSampler
    figure_vocabulary: FigureVocabulary
    duration_vocabulary: DurationVocabulary
    chord_vocabulary: ChordVocabularyConfig
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
        hand_entries = {
            hand: tuple(self.figure_vocabulary.filter(scale_type=scale_type, hand=hand).entries) for hand in _HANDS
        }
        motif_plan = (
            self._plan_motifs(
                form=form,
                slots=slots,
                anchors=anchors,
                chord_pitch_classes=chord_pitch_classes,
                hand_entries=hand_entries,
                shortest_note_duration=shortest_note_duration,
                scale_type=scale_type,
                bar_duration=bar_duration,
                rng=rng,
            )
            if self.config.lambda_similarity > 0.0
            else {}
        )

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
                        motif=motif_plan.get((hand, slot_index)),
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

    def _plan_motifs(
        self,
        *,
        form: FormTree,
        slots: tuple[RenderSlot, ...],
        anchors: dict[Hand, tuple[int, ...]],
        chord_pitch_classes: tuple[frozenset[int], ...],
        hand_entries: dict[Hand, tuple[FigureVocabularyEntry, ...]],
        shortest_note_duration: Fraction,
        scale_type: ScaleType,
        bar_duration: Fraction,
        rng: Generator,
    ) -> dict[tuple[Hand, int], _SlotMotif]:
        context = _MotifPlanContext(
            slots=slots,
            sound_slot_indices=tuple(
                index for index, slot in enumerate(slots) if slot.leaf_type is MetricalLeafType.SOUND
            ),
            chord_pitch_classes=chord_pitch_classes,
            shortest_note_duration=shortest_note_duration,
            scale_type=scale_type,
            bar_duration=bar_duration,
        )
        plan: dict[tuple[Hand, int], _SlotMotif] = {}
        for hand in _HANDS:
            self._plan_hand_motifs(
                plan, hand, form=form, context=context, anchors_hand=anchors[hand], entries=hand_entries[hand], rng=rng
            )

        return plan

    def _plan_hand_motifs(
        self,
        plan: dict[tuple[Hand, int], _SlotMotif],
        hand: Hand,
        *,
        form: FormTree,
        context: _MotifPlanContext,
        anchors_hand: tuple[int, ...],
        entries: tuple[FigureVocabularyEntry, ...],
        rng: Generator,
    ) -> None:
        schemas: dict[int, _ClassSchema] = {}
        for segment in form.segments:
            segment_indices = _segment_sound_indices(segment, context)
            if not segment_indices:
                continue

            stored = schemas.get(segment.class_label)
            if stored is not None and _is_restatement(stored, segment, segment_indices):
                self._plan_restatement(plan, hand, segment, stored.schema, segment_indices, anchors_hand, rng)
            else:
                self._plan_seed(
                    plan,
                    hand,
                    segment,
                    segment_indices,
                    schemas,
                    context=context,
                    anchors_hand=anchors_hand,
                    entries=entries,
                    rng=rng,
                )

    def _plan_seed(
        self,
        plan: dict[tuple[Hand, int], _SlotMotif],
        hand: Hand,
        segment: SegmentNode,
        segment_indices: list[int],
        schemas: dict[int, _ClassSchema],
        *,
        context: _MotifPlanContext,
        anchors_hand: tuple[int, ...],
        entries: tuple[FigureVocabularyEntry, ...],
        rng: Generator,
    ) -> None:
        schema = select_motif_seed(
            self._motif_slots(segment_indices, context=context, anchors_hand=anchors_hand, entries=entries),
            scale_type=context.scale_type,
            config=self.config,
            candidate_count=self.motif_config.seed_candidate_count,
            rng=rng,
        )
        if schema is None:
            return

        schemas[segment.class_label] = _ClassSchema(schema=schema, sound_slot_count=len(segment_indices))
        for motif_figure in schema.figures:
            plan[(hand, segment_indices[motif_figure.slot_index])] = _SlotMotif(figure=motif_figure.figure, reuse=False)

    def _motif_slots(
        self,
        segment_indices: list[int],
        *,
        context: _MotifPlanContext,
        anchors_hand: tuple[int, ...],
        entries: tuple[FigureVocabularyEntry, ...],
    ) -> list[MotifSlot]:
        motif_slots: list[MotifSlot] = []
        for local_index, slot_index in enumerate(segment_indices):
            slot = context.slots[slot_index]
            feasible = tuple(
                entry
                for entry in entries
                if figure_fits_slot(
                    entry.figure,
                    slot_duration=slot.duration,
                    shortest_note_duration=context.shortest_note_duration,
                    duration_vocabulary=self.duration_vocabulary,
                )
            )
            motif_slots.append(
                MotifSlot(
                    slot_index=local_index,
                    anchor=anchors_hand[slot_index],
                    target_slope=_target_slope(anchors_hand, slot_index),
                    chord_pitch_classes=context.chord_pitch_classes[slot_index],
                    weight=slot.weight,
                    entries=feasible,
                )
            )

        return motif_slots

    def _plan_restatement(
        self,
        plan: dict[tuple[Hand, int], _SlotMotif],
        hand: Hand,
        segment: SegmentNode,
        schema: MotifSchema,
        segment_indices: list[int],
        anchors_hand: tuple[int, ...],
        rng: Generator,
    ) -> None:
        varied = vary_motif(
            schema,
            segment.variation,
            variation_budget=self.motif_config.variation_budget,
            maximum_transpose=self.motif_config.maximum_transpose,
            rng=rng,
        )
        grounded = ground_motif(varied, base_anchor=anchors_hand[segment_indices[0]])
        for local_index, slot_index in enumerate(segment_indices):
            grounded_figure = grounded.get(local_index)
            if grounded_figure is not None:
                plan[(hand, slot_index)] = _SlotMotif(figure=grounded_figure.figure, reuse=True)

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
        motif: _SlotMotif | None,
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
                    motif=motif,
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
        motif: _SlotMotif | None,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        slot = slots[slot_index]
        anchor = anchors[slot_index]
        target_slope = _target_slope(anchors, slot_index)
        if motif is not None and not motif.reuse:
            applied, emitted = self._emit_figure(
                state, tokens, motif.figure, anchor=anchor, slot=slot, scale_type=scale_type
            )
            if applied is not None:
                return applied, emitted

        intended = motif.figure if motif is not None and motif.reuse else None
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
                intended=intended,
            )
            applied, emitted = self._emit_figure(
                state, tokens, entry.figure, anchor=anchor, slot=slot, scale_type=scale_type
            )
            if applied is not None:
                return applied, emitted

        return self._emit_or_rest(state, tokens, rest_tokens(slot.duration, self.duration_vocabulary), slot)

    def _emit_figure(
        self,
        state: GenerationConstraintState,
        tokens: list[Token],
        figure: FigureNGram,
        *,
        anchor: int,
        slot: RenderSlot,
        scale_type: ScaleType,
    ) -> tuple[GenerationConstraintState | None, list[Token]]:
        candidate = anchor_figure_to_tokens(
            figure=figure,
            anchor=anchor,
            base_duration=slot_base_duration(figure, slot.duration),
            scale_type=scale_type,
            duration_vocabulary=self.duration_vocabulary,
        )
        applied = self._try_apply(state, candidate)
        if applied is None:
            return None, tokens

        return applied, tokens + candidate

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
