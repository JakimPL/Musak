from collections import Counter

from musak_model.n_grams.profile.streaming.counting import count_sample_figure_signatures, sample_profile_payload
from musak_model.n_grams.profile.streaming.schema import FigureBatchResult, FigureBatchTask, FigureCountCounter
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def process_figure_batch_task(task: FigureBatchTask) -> FigureBatchResult:
    duration_vocabulary = DurationVocabulary(task.tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    counts: FigureCountCounter = Counter()
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
        sample_payloads.append((sample_index, sample_profile_payload(sample_index, sample.scale_type, sample_counts)))

    return FigureBatchResult(
        batch_index=task.batch_index,
        sample_start_index=task.sample_start_index,
        encoded_sample_count=len(task.encoded_lines),
        counts=counts,
        sample_payloads=tuple(sample_payloads),
    )
