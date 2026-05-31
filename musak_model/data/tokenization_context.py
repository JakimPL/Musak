from typing import Final

from musak_model.data.scale_matcher.schema import ScaleMatch
from musak_model.data.schema import ParsedScore, SpellingContextSource, TokenizationContext

_DEFAULT_SPELLING_KEY_FIFTHS: Final[int] = 0


def tokenization_context_from_scale_match(scale_match: ScaleMatch) -> TokenizationContext:
    spelling_key_fifths, spelling_source = _spelling_context_from_declared_key(
        scale_match.diagnostics.declared_key_fifths
    )
    return TokenizationContext(
        pitch_set_scale_root=scale_match.scale_root,
        pitch_set_scale_type=scale_match.scale_type,
        declared_key_fifths=scale_match.diagnostics.declared_key_fifths,
        spelling_key_fifths=spelling_key_fifths,
        spelling_context_source=spelling_source,
    )


def tokenization_context_from_score(score: ParsedScore) -> TokenizationContext:
    spelling_key_fifths, spelling_source = _spelling_context_from_score(score)
    return TokenizationContext(
        pitch_set_scale_root=score.scale_root,
        pitch_set_scale_type=score.scale_type,
        declared_key_fifths=score.declared_key_fifths,
        spelling_key_fifths=spelling_key_fifths,
        spelling_context_source=spelling_source,
    )


def _spelling_context_from_score(score: ParsedScore) -> tuple[int, SpellingContextSource]:
    if score.declared_key_fifths is not None:
        return score.key_fifths, SpellingContextSource.DECLARED_KEY_SIGNATURE

    if score.key_fifths != _DEFAULT_SPELLING_KEY_FIFTHS:
        return score.key_fifths, SpellingContextSource.SCORE_KEY_FIFTHS

    return _DEFAULT_SPELLING_KEY_FIFTHS, SpellingContextSource.DEFAULT_C_MAJOR


def _spelling_context_from_declared_key(
    declared_key_fifths: int | None,
) -> tuple[int, SpellingContextSource]:
    if declared_key_fifths is not None:
        return declared_key_fifths, SpellingContextSource.DECLARED_KEY_SIGNATURE

    return _DEFAULT_SPELLING_KEY_FIFTHS, SpellingContextSource.DEFAULT_C_MAJOR
