from dataclasses import dataclass

from musak_model.harmony.decoding.decoder import ViterbiChordDecoder
from musak_model.n_grams.profile.chord.schema import ChordDecodeSpec
from musak_model.synthetic.fitting.form.analysis import analyze_segment
from musak_model.synthetic.fitting.form.cadence import CadenceDetectionConfig, detect_cadences
from musak_model.synthetic.fitting.form.repetition import RepetitionConfig, analyze_repetition
from musak_model.synthetic.fitting.form.statistics import FormStatistics, accumulate_piece
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


@dataclass(frozen=True)
class FormBatchTask:
    batch_index: int
    sample_start_index: int
    encoded_lines: tuple[str, ...]
    tokenization_config: TokenizationConfig
    chord_decode: ChordDecodeSpec
    figure_min_n: int
    figure_max_n: int
    cadence_config: CadenceDetectionConfig
    repetition_config: RepetitionConfig


@dataclass(frozen=True)
class FormBatchResult:
    batch_index: int
    sample_start_index: int
    encoded_sample_count: int
    statistics: FormStatistics


def process_form_batch_task(task: FormBatchTask) -> FormBatchResult:
    duration_vocabulary = DurationVocabulary(task.tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    chord_decoder = ViterbiChordDecoder(config=task.chord_decode.decoder_config)
    statistics = FormStatistics()

    for line in task.encoded_lines:
        sample = EncodedExercise.model_validate_json(line)
        piece = analyze_segment(
            sample.to_segment(token_vocabulary=token_vocabulary),
            chord_decoder=chord_decoder,
            chord_vocabulary=task.chord_decode.vocabulary,
            duration_vocabulary=duration_vocabulary,
            figure_min_n=task.figure_min_n,
            figure_max_n=task.figure_max_n,
        )
        if piece is None:
            continue

        accumulate_piece(
            statistics,
            piece=piece,
            cadences=detect_cadences(piece, config=task.cadence_config),
            repetition=analyze_repetition(piece, config=task.repetition_config),
            similarity_bucket_count=task.repetition_config.similarity_bucket_count,
        )

    return FormBatchResult(
        batch_index=task.batch_index,
        sample_start_index=task.sample_start_index,
        encoded_sample_count=len(task.encoded_lines),
        statistics=statistics,
    )
