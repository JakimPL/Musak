from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from typing import TYPE_CHECKING

from musak_model.data.schema import Segment
from musak_model.n_grams.figure.parser import HandOnsetRun, PitchedOnset, extract_hand_onset_runs
from musak_model.n_grams.profile.rhythm.schema import RhythmCountCounter, RhythmCountKey
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_shared.ratios import format_ratio

if TYPE_CHECKING:
    from musak_model.training.ingestion.schema import EncodedExercise


def count_sample_rhythm_metrics(
    sample: EncodedExercise,
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    rhythm_min_n: int,
    rhythm_max_n: int,
    grid_alignment_denominators: tuple[int, ...],
    strong_beat_offsets: tuple[Fraction, ...],
) -> RhythmCountCounter:
    segment = sample.to_segment(token_vocabulary=token_vocabulary)
    return count_segment_rhythm_metrics(
        segment,
        duration_vocabulary=duration_vocabulary,
        rhythm_min_n=rhythm_min_n,
        rhythm_max_n=rhythm_max_n,
        grid_alignment_denominators=grid_alignment_denominators,
        strong_beat_offsets=strong_beat_offsets,
    )


def count_segment_rhythm_metrics(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    rhythm_min_n: int,
    rhythm_max_n: int,
    grid_alignment_denominators: tuple[int, ...],
    strong_beat_offsets: tuple[Fraction, ...],
) -> RhythmCountCounter:
    runs_by_hand = extract_hand_onset_runs(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
    )
    time_signature = f"{segment.time_numerator}/{segment.time_denominator}"
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    counts: RhythmCountCounter = Counter()
    for hand, runs in runs_by_hand.items():
        _count_duration_values(
            counts,
            runs=runs,
            scale_type=segment.scale_type.value,
            time_signature=time_signature,
            hand=hand,
        )
        _count_grid_alignment(
            counts,
            runs=runs,
            scale_type=segment.scale_type.value,
            time_signature=time_signature,
            hand=hand,
            grid_alignment_denominators=grid_alignment_denominators,
        )
        _count_strong_beat_onsets(
            counts,
            runs=runs,
            scale_type=segment.scale_type.value,
            time_signature=time_signature,
            hand=hand,
            measure_duration=measure_duration,
            strong_beat_offsets=strong_beat_offsets,
        )
        _count_onset_positions(
            counts,
            runs=runs,
            scale_type=segment.scale_type.value,
            time_signature=time_signature,
            hand=hand,
            measure_duration=measure_duration,
            grid_alignment_denominators=grid_alignment_denominators,
        )
        _count_bar_total(
            counts,
            scale_type=segment.scale_type.value,
            time_signature=time_signature,
            hand=hand,
            bar_count=segment.bar_count,
        )
        _count_rhythm_ngrams(
            counts,
            runs=runs,
            scale_type=segment.scale_type.value,
            time_signature=time_signature,
            hand=hand,
            rhythm_min_n=rhythm_min_n,
            rhythm_max_n=rhythm_max_n,
        )

    return counts


def _count_duration_values(
    counts: RhythmCountCounter,
    *,
    runs: tuple[HandOnsetRun, ...],
    scale_type: str,
    time_signature: str,
    hand: Hand,
) -> None:
    for onset in _iter_onsets(runs):
        counts[
            RhythmCountKey(
                scale_type=scale_type,
                time_signature=time_signature,
                hand=hand.value,
                kind="duration_value",
                parameter="",
                value=format_ratio(onset.duration),
            )
        ] += 1


def _count_grid_alignment(
    counts: RhythmCountCounter,
    *,
    runs: tuple[HandOnsetRun, ...],
    scale_type: str,
    time_signature: str,
    hand: Hand,
    grid_alignment_denominators: tuple[int, ...],
) -> None:
    for denominator in grid_alignment_denominators:
        grid = Fraction(1, denominator)
        parameter = str(denominator)
        for onset in _iter_onsets(runs):
            onset_value = _alignment_value(onset.start, grid=grid)
            duration_value = _alignment_value(onset.duration, grid=grid)
            onset_key = RhythmCountKey(
                scale_type=scale_type,
                time_signature=time_signature,
                hand=hand.value,
                kind="onset_grid_alignment",
                parameter=parameter,
                value=onset_value,
            )
            counts[onset_key] += 1
            duration_key = RhythmCountKey(
                scale_type=scale_type,
                time_signature=time_signature,
                hand=hand.value,
                kind="duration_grid_alignment",
                parameter=parameter,
                value=duration_value,
            )
            counts[duration_key] += 1


def _count_strong_beat_onsets(
    counts: RhythmCountCounter,
    *,
    runs: tuple[HandOnsetRun, ...],
    scale_type: str,
    time_signature: str,
    hand: Hand,
    measure_duration: Fraction,
    strong_beat_offsets: tuple[Fraction, ...],
) -> None:
    strong_offsets = frozenset(strong_beat_offsets)
    for onset in _iter_onsets(runs):
        beat_offset = onset.start % measure_duration
        value = "strong" if beat_offset in strong_offsets else "weak"
        key = RhythmCountKey(
            scale_type=scale_type,
            time_signature=time_signature,
            hand=hand.value,
            kind="strong_beat_onset",
            parameter="",
            value=value,
        )
        counts[key] += 1


def _count_onset_positions(
    counts: RhythmCountCounter,
    *,
    runs: tuple[HandOnsetRun, ...],
    scale_type: str,
    time_signature: str,
    hand: Hand,
    measure_duration: Fraction,
    grid_alignment_denominators: tuple[int, ...],
) -> None:
    onsets = _iter_onsets(runs)
    for denominator in grid_alignment_denominators:
        if not _grid_divides_measure(measure_duration, denominator):
            continue

        occupied_bars_per_cell = _cell_occupancy(onsets, measure_duration=measure_duration, denominator=denominator)
        for cell, occupied_bars in occupied_bars_per_cell.items():
            key = RhythmCountKey(
                scale_type=scale_type,
                time_signature=time_signature,
                hand=hand.value,
                kind="onset_position",
                parameter=str(denominator),
                value=str(cell),
            )
            counts[key] += occupied_bars


def _grid_divides_measure(measure_duration: Fraction, denominator: int) -> bool:
    return (measure_duration * denominator).denominator == 1


def _onset_bar_and_cell(onset: PitchedOnset, *, measure_duration: Fraction, denominator: int) -> tuple[int, int]:
    bar = int(onset.start // measure_duration)
    cell = int((onset.start % measure_duration) * denominator)
    return bar, cell


def _cell_occupancy(onsets: tuple[PitchedOnset, ...], *, measure_duration: Fraction, denominator: int) -> Counter[int]:
    occupied_bar_cells = {
        _onset_bar_and_cell(onset, measure_duration=measure_duration, denominator=denominator) for onset in onsets
    }
    occupied_bars_per_cell: Counter[int] = Counter()
    for _, cell in occupied_bar_cells:
        occupied_bars_per_cell[cell] += 1

    return occupied_bars_per_cell


def _count_bar_total(
    counts: RhythmCountCounter,
    *,
    scale_type: str,
    time_signature: str,
    hand: Hand,
    bar_count: int,
) -> None:
    key = RhythmCountKey(
        scale_type=scale_type,
        time_signature=time_signature,
        hand=hand.value,
        kind="bar_total",
        parameter="",
        value="",
    )
    counts[key] += bar_count


def _count_rhythm_ngrams(
    counts: RhythmCountCounter,
    *,
    runs: tuple[HandOnsetRun, ...],
    scale_type: str,
    time_signature: str,
    hand: Hand,
    rhythm_min_n: int,
    rhythm_max_n: int,
) -> None:
    for run in runs:
        onsets = run.onsets
        for n in range(rhythm_min_n, rhythm_max_n + 1):
            if len(onsets) < n:
                continue

            for start_index in range(0, len(onsets) - n + 1):
                window = onsets[start_index : start_index + n]
                key = RhythmCountKey(
                    scale_type=scale_type,
                    time_signature=time_signature,
                    hand=hand.value,
                    kind="rhythm_ngram",
                    parameter=str(n),
                    value=_rhythm_ngram_value(window),
                )
                counts[key] += 1


def _iter_onsets(runs: tuple[HandOnsetRun, ...]) -> tuple[PitchedOnset, ...]:
    return tuple(onset for run in runs for onset in run.onsets)


def _alignment_value(value: Fraction, *, grid: Fraction) -> str:
    return "aligned" if value % grid == 0 else "off_grid"


def _rhythm_ngram_value(onsets: tuple[PitchedOnset, ...]) -> str:
    return json.dumps(
        {
            "durations": [format_ratio(onset.duration) for onset in onsets],
            "iois": [format_ratio(onsets[index + 1].start - onsets[index].start) for index in range(len(onsets) - 1)],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
