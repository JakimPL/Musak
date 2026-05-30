from collections.abc import Sequence
from fractions import Fraction

from musak_model.harmony.decoding.candidates import Candidate
from musak_model.harmony.decoding.windows import SoundingWindow
from musak_model.harmony.schema import Chord


def viterbi_decode(
    windows: Sequence[SoundingWindow],
    *,
    candidates: Sequence[Candidate],
    self_transition_bias: float,
    non_chord_penalty: float,
) -> tuple[Chord, ...]:
    emissions = [
        [
            _emission_score(window.pitch_class_weights, candidate.pitch_classes, non_chord_penalty=non_chord_penalty)
            for candidate in candidates
        ]
        for window in windows
    ]
    best_scores = list(emissions[0])
    back_pointers: list[list[int]] = []
    for window_emissions in emissions[1:]:
        previous_scores = best_scores
        current_scores: list[float] = []
        current_pointers: list[int] = []
        for candidate_index, emission in enumerate(window_emissions):
            best_previous_index = _best_predecessor(
                previous_scores,
                candidate_index=candidate_index,
                self_transition_bias=self_transition_bias,
            )
            transition = self_transition_bias if best_previous_index == candidate_index else 0.0
            current_scores.append(emission + previous_scores[best_previous_index] + transition)
            current_pointers.append(best_previous_index)

        best_scores = current_scores
        back_pointers.append(current_pointers)

    return _backtrack(best_scores, back_pointers, candidates=candidates)


def _best_predecessor(
    previous_scores: Sequence[float],
    *,
    candidate_index: int,
    self_transition_bias: float,
) -> int:
    best_index = 0
    best_value = -float("inf")
    for previous_index, previous_score in enumerate(previous_scores):
        transition = self_transition_bias if previous_index == candidate_index else 0.0
        value = previous_score + transition
        if value > best_value:
            best_value = value
            best_index = previous_index

    return best_index


def _backtrack(
    best_scores: Sequence[float],
    back_pointers: Sequence[Sequence[int]],
    *,
    candidates: Sequence[Candidate],
) -> tuple[Chord, ...]:
    current_index = max(range(len(best_scores)), key=lambda index: best_scores[index])
    indices = [current_index]
    for pointers in reversed(back_pointers):
        current_index = pointers[current_index]
        indices.append(current_index)

    indices.reverse()
    return tuple(candidates[index].chord for index in indices)


def _emission_score(
    pitch_class_weights: dict[int, Fraction],
    chord_pitch_classes: frozenset[int],
    *,
    non_chord_penalty: float,
) -> float:
    coverage = 0.0
    leftover = 0.0
    for window_pitch_class, weight in pitch_class_weights.items():
        if window_pitch_class in chord_pitch_classes:
            coverage += float(weight)
        else:
            leftover += float(weight)

    return coverage - non_chord_penalty * leftover
