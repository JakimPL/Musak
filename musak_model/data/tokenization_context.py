from typing import Final

from musak_model.data.scale_matcher.schema import ScaleMatch
from musak_model.data.schema import ParsedScore, SpellingContextSource, TokenizationContext
from musak_model.tokens.schema import ScaleType
from musak_shared.elements import PITCHES_PER_OCTAVE, key_fifths_from_pitch_class

_DEFAULT_SPELLING_KEY_FIFTHS: Final[int] = 0
_PARENT_MAJOR_OFFSET_BY_SCALE_TYPE: Final[dict[ScaleType, int]] = {
    ScaleType.MAJOR: 0,
    ScaleType.HARMONIC_MINOR: -9,
    ScaleType.MELODIC_MINOR: -9,
}


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


def tokenization_context_from_scale(
    *,
    scale_root: int,
    scale_type: ScaleType,
) -> TokenizationContext:
    return TokenizationContext(
        pitch_set_scale_root=scale_root,
        pitch_set_scale_type=scale_type,
        declared_key_fifths=None,
        spelling_key_fifths=key_fifths_for_scale_basis(scale_root=scale_root, scale_type=scale_type),
        spelling_context_source=SpellingContextSource.PITCH_SET_BASIS,
    )


def key_fifths_for_scale_basis(*, scale_root: int, scale_type: ScaleType) -> int:
    parent_major_root = (scale_root + _PARENT_MAJOR_OFFSET_BY_SCALE_TYPE[scale_type]) % PITCHES_PER_OCTAVE
    return key_fifths_from_pitch_class(parent_major_root)


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
