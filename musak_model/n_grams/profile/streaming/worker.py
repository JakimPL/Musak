from collections import Counter

from musak_model.n_grams.profile.rhythm.extraction import RhythmCountCounter, count_sample_rhythm_metrics
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
    sample_payloads: list[tuple[int, str]] = []
    for sample_offset, line in enumerate(task.encoded_lines):
        sample_index = task.sample_start_index + sample_offset
        sample = EncodedExercise.model_validate_json(line)
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
        sample_payloads=tuple(sample_payloads),
    )
