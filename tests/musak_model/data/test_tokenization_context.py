from musak_model.data.schema import ParsedScore, SpellingContextSource
from musak_model.data.tokenization_context import tokenization_context_from_score
from musak_model.tokens.schema import ScaleType


def test_tokenization_context_from_score_uses_declared_key_for_spelling() -> None:
    context = tokenization_context_from_score(
        ParsedScore(
            declared_key_fifths=-1,
            scale_root=5,
            key_fifths=-1,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[],
            left_hand_bars=[],
        )
    )

    assert context.pitch_set_scale_root == 5
    assert context.pitch_set_scale_type == ScaleType.MAJOR
    assert context.declared_key_fifths == -1
    assert context.spelling_key_fifths == -1
    assert context.spelling_context_source == SpellingContextSource.DECLARED_KEY_SIGNATURE


def test_tokenization_context_distinguishes_declared_c_major_from_default_c_major() -> None:
    declared_context = tokenization_context_from_score(
        ParsedScore(
            declared_key_fifths=0,
            scale_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[],
            left_hand_bars=[],
        )
    )
    default_context = tokenization_context_from_score(
        ParsedScore(
            scale_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[],
            left_hand_bars=[],
        )
    )

    assert declared_context.declared_key_fifths == 0
    assert declared_context.spelling_key_fifths == 0
    assert declared_context.spelling_context_source == SpellingContextSource.DECLARED_KEY_SIGNATURE
    assert default_context.declared_key_fifths is None
    assert default_context.spelling_key_fifths == 0
    assert default_context.spelling_context_source == SpellingContextSource.DEFAULT_C_MAJOR


def test_tokenization_context_from_score_marks_nondefault_score_key_without_declared_key() -> None:
    context = tokenization_context_from_score(
        ParsedScore(
            scale_root=2,
            key_fifths=2,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[],
            left_hand_bars=[],
        )
    )

    assert context.declared_key_fifths is None
    assert context.spelling_key_fifths == 2
    assert context.spelling_context_source == SpellingContextSource.SCORE_KEY_FIFTHS


def test_tokenization_context_from_score_defaults_missing_spelling_to_c_major() -> None:
    context = tokenization_context_from_score(
        ParsedScore(
            scale_root=2,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[],
            left_hand_bars=[],
        )
    )

    assert context.declared_key_fifths is None
    assert context.spelling_key_fifths == 0
    assert context.spelling_context_source == SpellingContextSource.DEFAULT_C_MAJOR
