import json
from collections import Counter
from fractions import Fraction

from musak_model.n_grams.figure.parser import HandOnsetRun, PitchedOnset, extract_hand_onset_runs
from musak_model.n_grams.profile.rhythm.schema import RhythmCountKey
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise

type RhythmCountCounter = Counter[RhythmCountKey]


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
    tokens = token_vocabulary.decode(sample.token_ids)
    runs_by_hand = extract_hand_onset_runs(
        tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=sample.time_numerator,
        time_denominator=sample.time_denominator,
    )
    time_signature = f"{sample.time_numerator}/{sample.time_denominator}"
    measure_duration = Fraction(sample.time_numerator, sample.time_denominator)
    counts: RhythmCountCounter = Counter()
    for hand, runs in runs_by_hand.items():
        _count_duration_values(
            counts,
            runs=runs,
            scale_type=sample.scale_type.value,
            time_signature=time_signature,
            hand=hand,
        )
        _count_grid_alignment(
            counts,
            runs=runs,
            scale_type=sample.scale_type.value,
            time_signature=time_signature,
            hand=hand,
            grid_alignment_denominators=grid_alignment_denominators,
        )
        _count_strong_beat_onsets(
            counts,
            runs=runs,
            scale_type=sample.scale_type.value,
            time_signature=time_signature,
            hand=hand,
            measure_duration=measure_duration,
            strong_beat_offsets=strong_beat_offsets,
        )
        _count_rhythm_ngrams(
            counts,
            runs=runs,
            scale_type=sample.scale_type.value,
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
                value=_fraction_text(onset.duration),
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
            counts[
                RhythmCountKey(
                    scale_type=scale_type,
                    time_signature=time_signature,
                    hand=hand.value,
                    kind="onset_grid_alignment",
                    parameter=parameter,
                    value=onset_value,
                )
            ] += 1
            counts[
                RhythmCountKey(
                    scale_type=scale_type,
                    time_signature=time_signature,
                    hand=hand.value,
                    kind="duration_grid_alignment",
                    parameter=parameter,
                    value=duration_value,
                )
            ] += 1


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
        counts[
            RhythmCountKey(
                scale_type=scale_type,
                time_signature=time_signature,
                hand=hand.value,
                kind="strong_beat_onset",
                parameter="",
                value=value,
            )
        ] += 1


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
                counts[
                    RhythmCountKey(
                        scale_type=scale_type,
                        time_signature=time_signature,
                        hand=hand.value,
                        kind="rhythm_ngram",
                        parameter=str(n),
                        value=_rhythm_ngram_value(window),
                    )
                ] += 1


def _iter_onsets(runs: tuple[HandOnsetRun, ...]) -> tuple[PitchedOnset, ...]:
    return tuple(onset for run in runs for onset in run.onsets)


def _alignment_value(value: Fraction, *, grid: Fraction) -> str:
    return "aligned" if value % grid == 0 else "off_grid"


def _rhythm_ngram_value(onsets: tuple[PitchedOnset, ...]) -> str:
    return json.dumps(
        {
            "durations": [_fraction_text(onset.duration) for onset in onsets],
            "iois": [_fraction_text(onsets[index + 1].start - onsets[index].start) for index in range(len(onsets) - 1)],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"
