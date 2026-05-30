from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence

from musak_model.data.schema import Segment
from musak_model.harmony.decoding.decoder import ViterbiChordDecoder
from musak_model.harmony.decoding.schema import ChordWindow
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.n_grams.figure.signature import figure_signature_to_json, iter_figure_occurrences_from_run
from musak_model.n_grams.profile.chord.schema import (
    INITIAL_CHORD_SOURCE,
    ChordDecodeSpec,
    ChordStatistics,
    ChordTransitionCounts,
    ChordTransitionKey,
    FigureByChordCountKey,
    FigureByChordCounts,
    chord_to_key,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import scale_size_for_type


def chord_statistics(
    segments: Sequence[Segment],
    *,
    duration_vocabulary: DurationVocabulary,
    decode_spec: ChordDecodeSpec,
    min_n: int,
    max_n: int,
) -> ChordStatistics:
    decoder = ViterbiChordDecoder(config=decode_spec.decoder_config)
    transition_counts: ChordTransitionCounts = Counter()
    figure_by_chord_counts: FigureByChordCounts = Counter()
    for segment in segments:
        windows = decoder.decode(
            segment,
            duration_vocabulary=duration_vocabulary,
            vocabulary=decode_spec.vocabulary,
        )
        if not windows:
            continue

        _accumulate_transitions(transition_counts, windows, scale_type=segment.scale_type.value)
        _accumulate_figure_by_chord(
            figure_by_chord_counts,
            segment,
            windows=windows,
            duration_vocabulary=duration_vocabulary,
            min_n=min_n,
            max_n=max_n,
        )

    return ChordStatistics(transition_counts=transition_counts, figure_by_chord_counts=figure_by_chord_counts)


def _accumulate_transitions(
    transition_counts: ChordTransitionCounts,
    windows: Sequence[ChordWindow],
    *,
    scale_type: str,
) -> None:
    transition_counts[ChordTransitionKey(scale_type, INITIAL_CHORD_SOURCE, chord_to_key(windows[0].chord))] += 1
    for previous_window, current_window in zip(windows, windows[1:], strict=False):
        source = chord_to_key(previous_window.chord)
        destination = chord_to_key(current_window.chord)
        transition_counts[ChordTransitionKey(scale_type, source, destination)] += 1


def _accumulate_figure_by_chord(
    figure_by_chord_counts: FigureByChordCounts,
    segment: Segment,
    *,
    windows: Sequence[ChordWindow],
    duration_vocabulary: DurationVocabulary,
    min_n: int,
    max_n: int,
) -> None:
    scale_size = scale_size_for_type(segment.scale_type)
    window_starts = [window.start for window in windows]
    runs_by_hand = extract_hand_onset_runs(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
    )
    for hand, runs in runs_by_hand.items():
        for run in runs:
            for occurrence in iter_figure_occurrences_from_run(run, min_n=min_n, max_n=max_n, scale_size=scale_size):
                window = windows[max(0, bisect_right(window_starts, occurrence.start) - 1)]
                figure_by_chord_counts[
                    FigureByChordCountKey(
                        scale_type=segment.scale_type.value,
                        hand=hand.value,
                        figure_length=occurrence.figure_length,
                        chord=chord_to_key(window.chord),
                        figure=figure_signature_to_json(occurrence.signature),
                    )
                ] += 1
