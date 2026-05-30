from collections import Counter

from musak_model.data.schema import Segment
from musak_model.n_grams.profile.chord.extraction import chord_statistics
from musak_model.n_grams.profile.chord.schema import ChordStatistics
from musak_model.n_grams.profile.register.extraction import register_statistics
from musak_model.n_grams.profile.rhythm.extraction import count_sample_rhythm_metrics
from musak_model.n_grams.profile.rhythm.schema import RhythmCountCounter
from musak_model.n_grams.profile.streaming.counting import count_sample_figure_signatures, sample_profile_payload
from musak_model.n_grams.profile.streaming.schema import FigureBatchResult, FigureBatchTask, FigureCountCounter
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def process_figure_batch_task(task: FigureBatchTask) -> FigureBatchResult:
    duration_vocabulary = DurationVocabulary(task.tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    counts: FigureCountCounter = Counter()
    rhythm_counts: RhythmCountCounter = Counter()
    segments: list[Segment] = []
    sample_payloads: list[tuple[int, str]] = []
    for sample_offset, line in enumerate(task.encoded_lines):
        sample_index = task.sample_start_index + sample_offset
        sample = EncodedExercise.model_validate_json(line)
        segments.append(sample.to_segment(token_vocabulary=token_vocabulary))
        sample_counts = count_sample_figure_signatures(
            sample,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            min_n=task.min_n,
            max_n=task.max_n,
        )
        counts.update(sample_counts)
        rhythm_counts.update(
            count_sample_rhythm_metrics(
                sample,
                duration_vocabulary=duration_vocabulary,
                token_vocabulary=token_vocabulary,
                rhythm_min_n=task.rhythm_min_n,
                rhythm_max_n=task.rhythm_max_n,
                grid_alignment_denominators=task.grid_alignment_denominators,
                strong_beat_offsets=task.strong_beat_offsets,
            )
        )
        sample_payloads.append((sample_index, sample_profile_payload(sample_index, sample.scale_type, sample_counts)))

    return FigureBatchResult(
        batch_index=task.batch_index,
        sample_start_index=task.sample_start_index,
        encoded_sample_count=len(task.encoded_lines),
        counts=counts,
        rhythm_counts=rhythm_counts,
        register_statistics=register_statistics(
            segments,
            duration_vocabulary=duration_vocabulary,
            arch_basis_count=task.register_arch_basis_count,
        ),
        chord_statistics=_chord_statistics(task, segments, duration_vocabulary=duration_vocabulary),
        sample_payloads=tuple(sample_payloads),
    )


def _chord_statistics(
    task: FigureBatchTask,
    segments: list[Segment],
    *,
    duration_vocabulary: DurationVocabulary,
) -> ChordStatistics:
    if task.chord_decode is None:
        return ChordStatistics(transition_counts=Counter(), figure_by_chord_counts=Counter())

    return chord_statistics(
        segments,
        duration_vocabulary=duration_vocabulary,
        decode_spec=task.chord_decode,
        min_n=task.min_n,
        max_n=task.max_n,
    )
