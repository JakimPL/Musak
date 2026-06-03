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
from musak_model.synthetic.base_durations import BaseDurationDistribution, choose_base_duration
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry
from musak_model.synthetic.processes.accent import AccentFieldSampler, draw_onset_mask
from musak_model.synthetic.processes.density import RhythmicDensitySampler
from musak_model.synthetic.processes.pitch import RegisterCurveSampler
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.figure_selection import figure_span_units, select_figure
from musak_model.synthetic.render.motif import GroundedMotifFigure, MotifConfig, MotifFigure, MotifSchema, ground_motif
from musak_model.synthetic.render.slots import RenderSlot, render_slots
from musak_model.synthetic.render.variation import vary_motif
from musak_model.synthetic.structure.form import FormTree, SegmentNode
from musak_model.synthetic.structure.harmony_grammar import HarmonyGrammarSampler
from musak_model.synthetic.structure.meter import (
    MetricalLeafType,
    MetricalNode,
    MetricalTree,
    MetricalTreeSampler,
)
from musak_model.synthetic.substitution.emission import anchor_figure_to_tokens, rest_tokens
from musak_model.synthetic.substitution.scoring import figure_net_contour
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


def _segment_contains(segment: SegmentNode, offset: Fraction, bar_duration: Fraction) -> bool:
    return segment.start_bar * bar_duration <= offset < (segment.start_bar + segment.bar_span) * bar_duration


@dataclass(frozen=True)
class RenderedChord:
    offset: Fraction
    duration: Fraction
    chord: Chord


@dataclass(frozen=True)
class RenderResult:
    segment: Segment
    chords: tuple[RenderedChord, ...]


@dataclass(frozen=True)
class _ClassTexture:
    anchors: dict[Hand, tuple[int, ...]]
    density: dict[Hand, tuple[float, ...]]
    accent_weights: dict[Hand, tuple[float, ...]]
    onset_masks: dict[Hand, tuple[bool, ...]]


@dataclass
class _MotifWalk:
    base_anchor: int
    grounded: dict[int, GroundedMotifFigure] | None
    recording: list[MotifFigure]
    fire_index: int = 0


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
    accent_field_sampler: AccentFieldSampler
    grid_denominator: int
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
        return self.render_plan(
            time_numerator=time_numerator,
            time_denominator=time_denominator,
            scale_root=scale_root,
            scale_type=scale_type,
            form=form,
            harmonic_slot_duration=harmonic_slot_duration,
            constraints=constraints,
            source_file=source_file,
            rng=rng,
        ).segment

    def render_plan(
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
    ) -> RenderResult:
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

        grid_count_per_bar = self._grid_count_per_bar(bar_duration)
        cell_duration = Fraction(1, self.grid_denominator)
        chord_pitch_classes = tuple(
            chord_pitch_class_set(slot.chord, scale_type=scale_type, vocabulary=self.chord_vocabulary) for slot in slots
        )
        segment_slot_indices = [
            [index for index, slot in enumerate(slots) if _segment_contains(segment, slot.offset, bar_duration)]
            for segment in form.segments
        ]
        anchors, density, accent_weights, onset_masks = self._assemble_textures(
            form=form,
            segment_slot_indices=segment_slot_indices,
            scale_type=scale_type,
            grid_count_per_bar=grid_count_per_bar,
            rng=rng,
        )
        hand_entries = {
            hand: tuple(self.figure_vocabulary.filter(scale_type=scale_type, hand=hand).entries) for hand in _HANDS
        }

        state = GenerationConstraintState(constraints=constraints)
        tokens: list[Token] = []
        motif_enabled = self.config.lambda_similarity > 0.0
        class_motifs: dict[tuple[int, int], dict[Hand, MotifSchema]] = {}
        carried_pitches: dict[Hand, int | None] = {hand: None for hand in _HANDS}
        for segment_index, segment in enumerate(form.segments):
            class_key = (segment.class_label, segment.bar_span)
            motif_walks = self._segment_motif_walks(
                segment,
                class_key=class_key,
                slot_indices=segment_slot_indices[segment_index],
                anchors=anchors,
                class_motifs=class_motifs,
                motif_enabled=motif_enabled,
                rng=rng,
            )
            for bar_index in range(segment.start_bar, segment.start_bar + segment.bar_span):
                bar_slot_indices = [
                    index
                    for index, slot in enumerate(slots)
                    if bar_index * bar_duration <= slot.offset < (bar_index + 1) * bar_duration
                ]
                for hand in _HANDS:
                    state, tokens = self._append(state, tokens, HandToken(hand=hand))
                    state, tokens, carried_pitches[hand] = self._fill_hand_bar(
                        state=state,
                        tokens=tokens,
                        slots=slots,
                        bar_slot_indices=bar_slot_indices,
                        bar_index=bar_index,
                        hand=hand,
                        anchors=anchors[hand],
                        density=density[hand],
                        accent_weights=accent_weights[hand],
                        onset_mask=onset_masks[hand],
                        chord_pitch_classes=chord_pitch_classes,
                        scale_type=scale_type,
                        grid_count_per_bar=grid_count_per_bar,
                        cell_duration=cell_duration,
                        entries=hand_entries[hand],
                        motif_walk=motif_walks.get(hand),
                        carried_pitch=carried_pitches[hand],
                        rng=rng,
                    )
                state, tokens = self._append(state, tokens, BarToken())
            if motif_enabled and class_key not in class_motifs:
                class_motifs[class_key] = {
                    hand: MotifSchema(tuple(motif_walks[hand].recording))
                    for hand in _HANDS
                    if hand in motif_walks and motif_walks[hand].recording
                }

        state, tokens = self._append(state, tokens, EndToken())
        rendered_segment = Segment(
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
        rendered_chords = tuple(
            RenderedChord(offset=slot.offset, duration=slot.duration, chord=slot.chord) for slot in slots
        )
        return RenderResult(segment=rendered_segment, chords=rendered_chords)

    def _grid_count_per_bar(self, bar_duration: Fraction) -> int:
        cells = bar_duration * self.grid_denominator
        if cells.denominator != 1:
            raise ValueError(
                f"grid denominator {self.grid_denominator} does not tile a {bar_duration} bar into whole cells"
            )

        return int(cells)

    def _assemble_textures(
        self,
        *,
        form: FormTree,
        segment_slot_indices: list[list[int]],
        scale_type: ScaleType,
        grid_count_per_bar: int,
        rng: Generator,
    ) -> tuple[
        dict[Hand, tuple[int, ...]],
        dict[Hand, tuple[float, ...]],
        dict[Hand, tuple[float, ...]],
        dict[Hand, tuple[bool, ...]],
    ]:
        cache: dict[tuple[int, int], _ClassTexture] = {}
        anchors: dict[Hand, list[int]] = {hand: [] for hand in _HANDS}
        density: dict[Hand, list[float]] = {hand: [] for hand in _HANDS}
        accent_weights: dict[Hand, list[float]] = {hand: [] for hand in _HANDS}
        onset_masks: dict[Hand, list[bool]] = {hand: [] for hand in _HANDS}
        for segment, indices in zip(form.segments, segment_slot_indices, strict=True):
            class_key = (segment.class_label, segment.bar_span)
            texture = cache.get(class_key)
            if texture is None:
                texture = self._sample_class_texture(
                    scale_type=scale_type,
                    slot_count=len(indices),
                    bar_span=segment.bar_span,
                    grid_count_per_bar=grid_count_per_bar,
                    rng=rng,
                )
                cache[class_key] = texture

            for hand in _HANDS:
                anchors[hand].extend(texture.anchors[hand])
                density[hand].extend(texture.density[hand])
                accent_weights[hand].extend(texture.accent_weights[hand])
                onset_masks[hand].extend(texture.onset_masks[hand])

        return (
            {hand: tuple(anchors[hand]) for hand in _HANDS},
            {hand: tuple(density[hand]) for hand in _HANDS},
            {hand: tuple(accent_weights[hand]) for hand in _HANDS},
            {hand: tuple(onset_masks[hand]) for hand in _HANDS},
        )

    def _sample_class_texture(
        self,
        *,
        scale_type: ScaleType,
        slot_count: int,
        bar_span: int,
        grid_count_per_bar: int,
        rng: Generator,
    ) -> _ClassTexture:
        anchors = {
            hand: self.register_curve_sampler.sample(length=slot_count, scale_type=scale_type, hand=hand, rng=rng)
            for hand in _HANDS
        }
        density = {hand: self.rhythmic_density_sampler.sample(length=slot_count, rng=rng) for hand in _HANDS}
        accent_weights = {
            hand: self.accent_field_sampler.sample_weights(
                bar_count=bar_span,
                grid_count_per_bar=grid_count_per_bar,
                scale_type=scale_type,
                hand=hand,
                rng=rng,
            )
            for hand in _HANDS
        }
        onset_masks = {hand: draw_onset_mask(accent_weights[hand], rng=rng) for hand in _HANDS}
        return _ClassTexture(anchors=anchors, density=density, accent_weights=accent_weights, onset_masks=onset_masks)

    def _segment_motif_walks(
        self,
        segment: SegmentNode,
        *,
        class_key: tuple[int, int],
        slot_indices: list[int],
        anchors: dict[Hand, tuple[int, ...]],
        class_motifs: dict[tuple[int, int], dict[Hand, MotifSchema]],
        motif_enabled: bool,
        rng: Generator,
    ) -> dict[Hand, _MotifWalk]:
        if not motif_enabled:
            return {}

        recorded = class_motifs.get(class_key)
        walks: dict[Hand, _MotifWalk] = {}
        for hand in _HANDS:
            base_anchor = anchors[hand][slot_indices[0]]
            grounded: dict[int, GroundedMotifFigure] | None = None
            if recorded is not None and hand in recorded:
                varied = vary_motif(
                    recorded[hand],
                    segment.variation,
                    variation_budget=self.motif_config.variation_budget,
                    maximum_transpose=self.motif_config.maximum_transpose,
                    rng=rng,
                )
                grounded = ground_motif(varied, base_anchor=base_anchor)

            walks[hand] = _MotifWalk(base_anchor=base_anchor, grounded=grounded, recording=[])

        return walks

    def _fill_hand_bar(
        self,
        *,
        state: GenerationConstraintState,
        tokens: list[Token],
        slots: tuple[RenderSlot, ...],
        bar_slot_indices: list[int],
        bar_index: int,
        hand: Hand,
        anchors: tuple[int, ...],
        density: tuple[float, ...],
        accent_weights: tuple[float, ...],
        onset_mask: tuple[bool, ...],
        chord_pitch_classes: tuple[frozenset[int], ...],
        scale_type: ScaleType,
        grid_count_per_bar: int,
        cell_duration: Fraction,
        entries: tuple[FigureVocabularyEntry, ...],
        motif_walk: _MotifWalk | None,
        carried_pitch: int | None,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token], int | None]:
        bar_start = state.constraints.bar_start(bar_index)
        bar_end = state.constraints.bar_end(bar_index)
        while state.cursor(hand) < bar_end:
            cursor = state.cursor(hand)
            fired_position = self._next_fired_position(
                onset_mask=onset_mask,
                slots=slots,
                bar_slot_indices=bar_slot_indices,
                bar_index=bar_index,
                grid_count_per_bar=grid_count_per_bar,
                cell_duration=cell_duration,
                bar_start=bar_start,
                cursor=cursor,
            )
            if fired_position is None:
                state, tokens = self._emit_rest(state, tokens, bar_end - cursor)
                return state, tokens, carried_pitch

            fired_offset = bar_start + fired_position * cell_duration
            if fired_offset > cursor:
                state, tokens = self._emit_rest(state, tokens, fired_offset - cursor)
                continue

            slot_index = self._slot_index_at(slots, bar_slot_indices, fired_offset)
            remaining = bar_end - cursor
            feasible = self._feasible_entries(
                entries, hand=hand, scale_type=scale_type, density_offset=density[slot_index], remaining=remaining
            )
            if not feasible:
                state, tokens = self._emit_rest(state, tokens, remaining)
                return state, tokens, carried_pitch

            cell_index = bar_index * grid_count_per_bar + fired_position
            register_anchor = anchors[slot_index]
            continuity_anchor = self._continuity_anchor(register_anchor, carried_pitch)
            anchor, intended = self._motif_anchor_and_intent(motif_walk, default_anchor=continuity_anchor)
            target_slope = (register_anchor - anchor) + _target_slope(anchors, slot_index)
            state, tokens, consumed, entry = self._place_one_figure(
                state=state,
                tokens=tokens,
                feasible=feasible,
                hand=hand,
                anchor=anchor,
                target_slope=target_slope,
                density_offset=density[slot_index],
                remaining=remaining,
                weight=accent_weights[cell_index],
                chord_pitch_classes=chord_pitch_classes[slot_index],
                scale_type=scale_type,
                intended=intended,
                metrical_position=fired_position,
                grid_count_per_bar=grid_count_per_bar,
                rng=rng,
            )
            if consumed is None or entry is None:
                state, tokens = self._emit_rest(state, tokens, min(cell_duration, bar_end - state.cursor(hand)))
            else:
                carried_pitch = anchor + figure_net_contour(entry.figure)
                if motif_walk is not None:
                    self._record_motif_figure(motif_walk, entry=entry, anchor=anchor)

        return state, tokens, carried_pitch

    def _continuity_anchor(self, register_anchor: int, carried_pitch: int | None) -> int:
        if carried_pitch is None:
            return register_anchor

        return round(register_anchor + self.config.melodic_continuity * (carried_pitch - register_anchor))

    @staticmethod
    def _motif_anchor_and_intent(
        motif_walk: _MotifWalk | None, *, default_anchor: int
    ) -> tuple[int, FigureNGram | None]:
        if motif_walk is None or motif_walk.grounded is None:
            return default_anchor, None

        grounded_figure = motif_walk.grounded.get(motif_walk.fire_index)
        if grounded_figure is None:
            return default_anchor, None

        return grounded_figure.anchor, grounded_figure.figure

    @staticmethod
    def _record_motif_figure(motif_walk: _MotifWalk, *, entry: FigureVocabularyEntry, anchor: int) -> None:
        if motif_walk.grounded is None:
            motif_walk.recording.append(
                MotifFigure(
                    slot_index=motif_walk.fire_index,
                    figure=entry.figure,
                    anchor_offset=anchor - motif_walk.base_anchor,
                )
            )

        motif_walk.fire_index += 1

    def _next_fired_position(
        self,
        *,
        onset_mask: tuple[bool, ...],
        slots: tuple[RenderSlot, ...],
        bar_slot_indices: list[int],
        bar_index: int,
        grid_count_per_bar: int,
        cell_duration: Fraction,
        bar_start: Fraction,
        cursor: Fraction,
    ) -> int | None:
        start_position = max(0, int((cursor - bar_start) // cell_duration))
        for position in range(start_position, grid_count_per_bar):
            onset_time = bar_start + position * cell_duration
            if onset_time < cursor:
                continue

            slot_index = self._slot_index_at(slots, bar_slot_indices, onset_time)
            slot_is_sound = slots[slot_index].leaf_type is MetricalLeafType.SOUND
            if onset_mask[bar_index * grid_count_per_bar + position] and slot_is_sound:
                return position

        return None

    def _slot_index_at(self, slots: tuple[RenderSlot, ...], bar_slot_indices: list[int], offset: Fraction) -> int:
        for slot_index in bar_slot_indices:
            slot = slots[slot_index]
            if slot.offset <= offset < slot.offset + slot.duration:
                return slot_index

        return bar_slot_indices[-1]

    def _feasible_entries(
        self,
        entries: tuple[FigureVocabularyEntry, ...],
        *,
        hand: Hand,
        scale_type: ScaleType,
        density_offset: float,
        remaining: Fraction,
    ) -> tuple[FigureVocabularyEntry, ...]:
        return tuple(
            entry
            for entry in entries
            if self._fitting_base(
                entry, hand=hand, scale_type=scale_type, density_offset=density_offset, remaining=remaining
            )
            is not None
        )

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
        intended: FigureNGram | None,
        metrical_position: int,
        grid_count_per_bar: int,
        rng: Generator,
    ) -> tuple[GenerationConstraintState, list[Token], Fraction | None, FigureVocabularyEntry | None]:
        for _ in range(self.config.max_resample_retries):
            entry = select_figure(
                feasible,
                anchor=anchor,
                target_slope=target_slope,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
                weight=weight,
                config=self.config,
                intended=intended,
                metrical_position=metrical_position,
                grid_count_per_bar=grid_count_per_bar,
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
                return applied, tokens + candidate, figure_span_units(entry.figure) * base_duration, entry

        return state, tokens, None, None

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

    def _emit_rest(
        self,
        state: GenerationConstraintState,
        tokens: list[Token],
        duration: Fraction,
    ) -> tuple[GenerationConstraintState, list[Token]]:
        if duration <= 0:
            return state, tokens

        rest = rest_tokens(duration, self.duration_vocabulary)
        applied = self._try_apply(state, rest) if rest is not None else None
        if applied is None:
            raise GenerationConstraintError(f"could not fill a {duration} rest with vocabulary durations")

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
