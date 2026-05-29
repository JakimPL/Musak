from collections.abc import Sequence
from fractions import Fraction
from typing import Final

import numpy as np

from musak_model.data.schema import Segment
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_token_to_midi_pitch
from musak_model.tokens.schema import Hand
from musak_shared.elements import PITCHES_PER_OCTAVE

_METRIC_PREFIX: Final[str] = "musical"
_CONSONANT_INTERVAL_CLASSES: Final[frozenset[int]] = frozenset({0, 3, 4, 5, 7, 8, 9})
_MINIMUM_AUTOCORRELATION_LENGTH: Final[int] = 3

type OnsetPitches = list[tuple[Fraction, tuple[int, ...]]]


def musical_metrics(
    segments: Sequence[Segment],
    *,
    duration_vocabulary: DurationVocabulary,
    metric_prefix: str = _METRIC_PREFIX,
) -> dict[str, float]:
    consonant_pairs = 0
    coincident_pairs = 0
    autocorrelations: list[float] = []
    for segment in segments:
        onset_pitches = _onset_pitches_by_hand(segment, duration_vocabulary=duration_vocabulary)
        consonant, total = _coincident_consonance_counts(onset_pitches)
        consonant_pairs += consonant
        coincident_pairs += total
        autocorrelations.extend(_register_autocorrelations(onset_pitches))

    metrics: dict[str, float] = {f"{metric_prefix}/count/coincident_onset_pairs": float(coincident_pairs)}
    if coincident_pairs > 0:
        metrics[f"{metric_prefix}/rate/harmonic_consonance"] = consonant_pairs / coincident_pairs

    if autocorrelations:
        metrics[f"{metric_prefix}/mean/register_lag1_autocorrelation"] = float(np.mean(autocorrelations))

    return metrics


def _onset_pitches_by_hand(segment: Segment, *, duration_vocabulary: DurationVocabulary) -> dict[Hand, OnsetPitches]:
    runs_by_hand = extract_hand_onset_runs(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
    )
    return {
        hand: [
            (
                onset.start,
                tuple(
                    note_token_to_midi_pitch(
                        note, scale_root=segment.scale_root, scale_type=segment.scale_type, hand=hand
                    )
                    for note in onset.notes
                ),
            )
            for run in runs
            for onset in run.onsets
        ]
        for hand, runs in runs_by_hand.items()
    }


def _coincident_consonance_counts(onset_pitches: dict[Hand, OnsetPitches]) -> tuple[int, int]:
    left_pitches_by_start = dict(onset_pitches.get(Hand.LEFT, []))
    consonant = 0
    total = 0
    for start, right_pitches in onset_pitches.get(Hand.RIGHT, []):
        left_pitches = left_pitches_by_start.get(start)
        if left_pitches is None:
            continue

        for right_pitch in right_pitches:
            for left_pitch in left_pitches:
                total += 1
                if abs(right_pitch - left_pitch) % PITCHES_PER_OCTAVE in _CONSONANT_INTERVAL_CLASSES:
                    consonant += 1

    return consonant, total


def _register_autocorrelations(onset_pitches: dict[Hand, OnsetPitches]) -> list[float]:
    autocorrelations: list[float] = []
    for onsets in onset_pitches.values():
        lowest_pitches = [min(pitches) for _, pitches in onsets if pitches]
        autocorrelation = _lag1_autocorrelation(lowest_pitches)
        if autocorrelation is not None:
            autocorrelations.append(autocorrelation)

    return autocorrelations


def _lag1_autocorrelation(values: Sequence[int]) -> float | None:
    if len(values) < _MINIMUM_AUTOCORRELATION_LENGTH:
        return None

    series = np.asarray(values, dtype=np.float64)
    current = series[:-1]
    following = series[1:]
    if current.std() == 0.0 or following.std() == 0.0:
        return None

    return float(np.corrcoef(current, following)[0, 1])
