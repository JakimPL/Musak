from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import torch
from torch import Tensor

from musak_model.conditioning.harmony.alignment import harmonic_plan_windows_from_decoder_coordinates
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.generation.constraints import GenerationConstraints
from musak_model.generation.coordinates import DecoderInputCoordinates
from musak_model.harmony.expansion import UnspellableChordError, chord_pitch_class_set, expand_chord_to_tones
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.pitch import degree_pitch_class
from musak_model.tokens.schema import NoteToken, ScaleType, Token
from musak_shared.elements import PITCHES_PER_OCTAVE

HARMONIC_RELATION_IGNORE_ID: Final[int] = -1
CHORD_ROOT_MEMBER_INDEX: Final[int] = 0
CHORD_THIRD_MEMBER_INDEX: Final[int] = 1
CHORD_FIFTH_MEMBER_INDEX: Final[int] = 2
CHROMATIC_NEIGHBOR_DISTANCE: Final[int] = 1


class HarmonicRelationId(IntEnum):
    CHORD_ROOT = 0
    CHORD_THIRD = 1
    CHORD_FIFTH = 2
    CHORD_SEVENTH = 3
    DIATONIC_NON_CHORD = 4
    CHROMATIC_NEIGHBOR = 5
    OTHER_CHROMATIC = 6


HARMONIC_RELATION_CLASS_COUNT: Final[int] = len(HarmonicRelationId)


@dataclass(frozen=True)
class HarmonicRelationTargetTensors:
    relation_ids: Tensor

    def to(self, device: torch.device) -> HarmonicRelationTargetTensors:
        return HarmonicRelationTargetTensors(relation_ids=self.relation_ids.to(device))


def harmonic_relation_target_tensors_from_tokens(
    tokens: Sequence[Token],
    *,
    windows: Sequence[HarmonicPlanWindow],
    constraints: GenerationConstraints,
    coordinates: DecoderInputCoordinates,
    scale_type: ScaleType,
    chord_vocabulary: ChordVocabularyConfig,
    duration_tick_denominator: int,
) -> HarmonicRelationTargetTensors:
    aligned_windows = harmonic_plan_windows_from_decoder_coordinates(
        windows,
        constraints=constraints,
        coordinates=coordinates,
        duration_tick_denominator=duration_tick_denominator,
    )
    if len(tokens) != len(aligned_windows):
        raise ValueError("token and harmonic-plan alignment lengths must match")

    return HarmonicRelationTargetTensors(
        relation_ids=torch.tensor(
            [
                harmonic_relation_id_for_token(
                    token,
                    window=window,
                    scale_type=scale_type,
                    chord_vocabulary=chord_vocabulary,
                )
                for token, window in zip(tokens, aligned_windows, strict=True)
            ],
            dtype=torch.long,
        )
    )


def harmonic_relation_id_for_token(
    token: Token,
    *,
    window: HarmonicPlanWindow | None,
    scale_type: ScaleType,
    chord_vocabulary: ChordVocabularyConfig,
) -> int:
    if not isinstance(token, NoteToken) or window is None:
        return HARMONIC_RELATION_IGNORE_ID

    try:
        return int(
            _harmonic_relation_id_for_note(
                token,
                window=window,
                scale_type=scale_type,
                chord_vocabulary=chord_vocabulary,
            )
        )
    except UnspellableChordError:
        return HARMONIC_RELATION_IGNORE_ID


def _harmonic_relation_id_for_note(
    note: NoteToken,
    *,
    window: HarmonicPlanWindow,
    scale_type: ScaleType,
    chord_vocabulary: ChordVocabularyConfig,
) -> HarmonicRelationId:
    note_pitch_class = degree_pitch_class(note.degree, note.accidental, scale_type=scale_type)
    chord_tones = expand_chord_to_tones(window.chord, scale_type=scale_type, vocabulary=chord_vocabulary)
    for member_index, chord_tone in enumerate(chord_tones):
        chord_tone_pitch_class = degree_pitch_class(chord_tone.degree, chord_tone.accidental, scale_type=scale_type)
        if note_pitch_class == chord_tone_pitch_class:
            return _chord_member_relation_id(member_index)

    if note.accidental == 0:
        return HarmonicRelationId.DIATONIC_NON_CHORD

    chord_pitch_classes = chord_pitch_class_set(window.chord, scale_type=scale_type, vocabulary=chord_vocabulary)
    if any(
        _pitch_class_distance(note_pitch_class, chord_pitch_class) == CHROMATIC_NEIGHBOR_DISTANCE
        for chord_pitch_class in chord_pitch_classes
    ):
        return HarmonicRelationId.CHROMATIC_NEIGHBOR

    return HarmonicRelationId.OTHER_CHROMATIC


def _chord_member_relation_id(member_index: int) -> HarmonicRelationId:
    if member_index == CHORD_ROOT_MEMBER_INDEX:
        return HarmonicRelationId.CHORD_ROOT
    if member_index == CHORD_THIRD_MEMBER_INDEX:
        return HarmonicRelationId.CHORD_THIRD
    if member_index == CHORD_FIFTH_MEMBER_INDEX:
        return HarmonicRelationId.CHORD_FIFTH

    return HarmonicRelationId.CHORD_SEVENTH


def _pitch_class_distance(left_pitch_class: int, right_pitch_class: int) -> int:
    distance = abs(left_pitch_class - right_pitch_class) % PITCHES_PER_OCTAVE
    return min(distance, PITCHES_PER_OCTAVE - distance)
