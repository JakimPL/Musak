from collections.abc import Mapping, Sequence

import numpy as np

from musak_model.data.schema import Segment
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.n_grams.profile.register.dct import trend_and_residual
from musak_model.n_grams.profile.register.schema import RegisterStatistics, RegisterStatisticsKey, RegisterSums
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_diatonic_position
from musak_model.tokens.schema import scale_size_for_type

type RegisterSequences = Mapping[RegisterStatisticsKey, Sequence[Sequence[float]]]


def register_statistics(
    segments: Sequence[Segment],
    *,
    duration_vocabulary: DurationVocabulary,
    arch_basis_count: int,
) -> RegisterStatistics:
    sequences_by_group: dict[RegisterStatisticsKey, list[Sequence[float]]] = {}
    for segment in segments:
        scale_size = scale_size_for_type(segment.scale_type)
        runs_by_hand = extract_hand_onset_runs(
            segment.tokens,
            duration_vocabulary=duration_vocabulary,
            time_numerator=segment.time_numerator,
            time_denominator=segment.time_denominator,
        )
        for hand, runs in runs_by_hand.items():
            sequence = [
                float(min(note_diatonic_position(note, scale_size=scale_size) for note in onset.notes))
                for run in runs
                for onset in run.onsets
            ]
            if sequence:
                key = RegisterStatisticsKey(scale_type=segment.scale_type.value, hand=hand.value)
                sequences_by_group.setdefault(key, []).append(sequence)

    return register_statistics_from_sequences(sequences_by_group, arch_basis_count=arch_basis_count)


def register_statistics_from_sequences(
    sequences_by_group: RegisterSequences,
    *,
    arch_basis_count: int,
) -> RegisterStatistics:
    statistics: RegisterStatistics = {}
    minimum_length = arch_basis_count + 2
    for group, sequences in sequences_by_group.items():
        trend_square_sum = 0.0
        residual_square_sum = 0.0
        residual_lag_product_sum = 0.0
        element_count = 0
        for sequence in sequences:
            values = np.asarray(sequence, dtype=np.float64)
            if values.size < minimum_length:
                continue

            trend, residual = trend_and_residual(values - values.mean(), arch_basis_count=arch_basis_count)
            trend_square_sum += float(trend @ trend)
            residual_square_sum += float(residual @ residual)
            residual_lag_product_sum += float(residual[:-1] @ residual[1:])
            element_count += residual.size

        if element_count == 0:
            continue

        statistics[group] = RegisterSums(
            trend_square_sum=trend_square_sum,
            residual_square_sum=residual_square_sum,
            residual_lag_product_sum=residual_lag_product_sum,
            element_count=element_count,
        )

    return statistics
