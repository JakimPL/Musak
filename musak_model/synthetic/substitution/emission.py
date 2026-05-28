from fractions import Fraction

from musak_model.n_grams.figure.builder import scale_size_for_type
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import JoinWithPreviousToken, NoteToken, ScaleType, Token


def anchor_figure_to_tokens(
    *,
    figure: FigureNGram,
    anchor: int,
    base_duration: Fraction,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    scale_size = scale_size_for_type(scale_type)
    tokens: list[Token] = []
    for degrees, normalized_duration in figure.onsets:
        absolute_duration = normalized_duration * base_duration
        duration_id = duration_vocabulary.require_duration_id(absolute_duration)
        for note_index, (relative_position, accidental) in enumerate(degrees):
            absolute_position = anchor + relative_position
            absolute_degree = absolute_position % scale_size + 1
            octave_offset = absolute_position // scale_size
            tokens.append(
                NoteToken(
                    degree=absolute_degree,
                    accidental=accidental,
                    octave_offset=octave_offset,
                    duration_id=duration_id,
                )
            )
            if note_index > 0:
                tokens.append(JoinWithPreviousToken())

    return tokens
