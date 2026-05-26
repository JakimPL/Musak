from musak_model.n_grams.figure.builder import scale_size_for_type
from musak_model.n_grams.figure.counter import FigureNGramCountsByHand, count_hand_figure_ngrams
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def count_encoded_exercise_figure_ngrams(
    sample: EncodedExercise,
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
) -> FigureNGramCountsByHand:
    tokens = token_vocabulary.decode(sample.token_ids)
    runs_by_hand = extract_hand_onset_runs(
        tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=sample.time_numerator,
        time_denominator=sample.time_denominator,
    )
    return count_hand_figure_ngrams(
        runs_by_hand,
        min_n=min_n,
        max_n=max_n,
        scale_size=scale_size_for_type(sample.scale_type),
    )
