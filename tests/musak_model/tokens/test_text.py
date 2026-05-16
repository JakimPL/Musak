from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import pytest

from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    StartToken,
    Token,
)
from musak_model.tokens.text import (
    TokenTextParseError,
    UnsupportedTokenDurationError,
    token_from_text,
    tokens_from_text,
    tokens_to_text,
)


def _duration_id(duration_vocabulary: DurationVocabulary, duration: Fraction) -> int:
    return duration_vocabulary.fraction_to_id(duration)


def _note(
    duration_vocabulary: DurationVocabulary,
    *,
    degree: int,
    duration: Fraction,
    accidental: int = 0,
    octave_offset: int = 0,
) -> NoteToken:
    return NoteToken(
        degree=degree,
        accidental=accidental,
        octave_offset=octave_offset,
        duration_id=_duration_id(duration_vocabulary, duration),
    )


def _rest(duration_vocabulary: DurationVocabulary, duration: Fraction) -> RestToken:
    return RestToken(duration_id=_duration_id(duration_vocabulary, duration))


def _hold(duration_vocabulary: DurationVocabulary, duration: Fraction) -> HoldToken:
    return HoldToken(duration_id=_duration_id(duration_vocabulary, duration))


def _canonical_sequence(duration_vocabulary: DurationVocabulary) -> list[Token]:
    return [
        HandToken(hand=Hand.RIGHT),
        _note(duration_vocabulary, degree=6, accidental=1, octave_offset=1, duration=Fraction(1, 4)),
        _note(duration_vocabulary, degree=3, duration=Fraction(1, 4)),
        JoinWithPreviousToken(),
        HandToken(hand=Hand.LEFT),
        _rest(duration_vocabulary, Fraction(1, 8)),
        _note(duration_vocabulary, degree=1, octave_offset=-1, duration=Fraction(1, 8)),
        BarToken(),
        HandToken(hand=Hand.RIGHT),
        _hold(duration_vocabulary, Fraction(1, 4)),
        EndToken(),
    ]


TokenFactory = Callable[[DurationVocabulary], Token]
TokenSequenceFactory = Callable[[DurationVocabulary], list[Token]]


@dataclass(frozen=True)
class TokenDisplayCase:
    name: str
    token_factory: TokenFactory
    expected_text: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TokenParseCase:
    name: str
    text: str
    expected_factory: TokenFactory

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TokenSequenceCase:
    name: str
    text: str
    expected_factory: TokenSequenceFactory

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TokenParseErrorCase:
    name: str
    text: str
    expected_error: type[TokenTextParseError]
    expected_match: str

    def __str__(self) -> str:
        return self.name


class TestTokenTextDisplay:
    CASES = [
        TokenDisplayCase(
            name="sharp note with positive octave offset",
            token_factory=lambda vocabulary: _note(
                vocabulary,
                degree=6,
                accidental=1,
                octave_offset=1,
                duration=Fraction(1, 4),
            ),
            expected_text="6♯↑1(1:4)",
        ),
        TokenDisplayCase(
            name="flat note with negative octave offset",
            token_factory=lambda vocabulary: _note(
                vocabulary,
                degree=2,
                accidental=-1,
                octave_offset=-2,
                duration=Fraction(1, 8),
            ),
            expected_text="2♭↓2(1:8)",
        ),
        TokenDisplayCase(
            name="natural whole note",
            token_factory=lambda vocabulary: _note(vocabulary, degree=1, duration=Fraction(1, 1)),
            expected_text="1(1:1)",
        ),
        TokenDisplayCase(
            name="dotted duration rest",
            token_factory=lambda vocabulary: _rest(vocabulary, Fraction(3, 8)),
            expected_text="r(3:8)",
        ),
        TokenDisplayCase(
            name="hold duration",
            token_factory=lambda vocabulary: _hold(vocabulary, Fraction(1, 4)),
            expected_text="h(1:4)",
        ),
        TokenDisplayCase(
            name="tuplet duration note",
            token_factory=lambda vocabulary: _note(vocabulary, degree=5, duration=Fraction(1, 12)),
            expected_text="5(1:12)",
        ),
        TokenDisplayCase(
            name="right hand",
            token_factory=lambda vocabulary: HandToken(hand=Hand.RIGHT),
            expected_text="R",
        ),
        TokenDisplayCase(
            name="left hand",
            token_factory=lambda vocabulary: HandToken(hand=Hand.LEFT),
            expected_text="L",
        ),
        TokenDisplayCase(
            name="join previous",
            token_factory=lambda vocabulary: JoinWithPreviousToken(),
            expected_text="~",
        ),
        TokenDisplayCase(
            name="bar",
            token_factory=lambda vocabulary: BarToken(),
            expected_text="|",
        ),
        TokenDisplayCase(
            name="start",
            token_factory=lambda vocabulary: StartToken(),
            expected_text="BOS",
        ),
        TokenDisplayCase(
            name="end",
            token_factory=lambda vocabulary: EndToken(),
            expected_text="‖",
        ),
    ]

    @pytest.mark.parametrize("case", CASES, ids=str)
    def test_token_to_text_uses_canonical_compact_format(
        self,
        case: TokenDisplayCase,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        token = case.token_factory(duration_vocabulary)

        assert token.to_text(duration_vocabulary=duration_vocabulary) == case.expected_text


class TestTokenTextDumping:
    CASES = [
        TokenSequenceCase(
            name="mixed unified stream",
            text="R 6♯↑1(1:4) 3(1:4) ~ L r(1:8) 1↓1(1:8) | R h(1:4) ‖",
            expected_factory=_canonical_sequence,
        ),
        TokenSequenceCase(
            name="dotted and tuplet durations",
            text="R 1(3:8) 5(1:12) ‖",
            expected_factory=lambda vocabulary: [
                HandToken(hand=Hand.RIGHT),
                _note(vocabulary, degree=1, duration=Fraction(3, 8)),
                _note(vocabulary, degree=5, duration=Fraction(1, 12)),
                EndToken(),
            ],
        ),
    ]

    @pytest.mark.parametrize("case", CASES, ids=str)
    def test_tokens_to_text_dumps_space_separated_canonical_sequence(
        self,
        case: TokenSequenceCase,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        tokens = case.expected_factory(duration_vocabulary)

        assert tokens_to_text(tokens, duration_vocabulary=duration_vocabulary) == case.text


class TestTokenTextParsing:
    TOKEN_CASES = [
        TokenParseCase(
            name="canonical sharp note",
            text="6♯↑1(1:4)",
            expected_factory=lambda vocabulary: _note(
                vocabulary,
                degree=6,
                accidental=1,
                octave_offset=1,
                duration=Fraction(1, 4),
            ),
        ),
        TokenParseCase(
            name="canonical flat note",
            text="2♭↓1(1:8)",
            expected_factory=lambda vocabulary: _note(
                vocabulary,
                degree=2,
                accidental=-1,
                octave_offset=-1,
                duration=Fraction(1, 8),
            ),
        ),
        TokenParseCase(
            name="ascii sharp alias",
            text="6#↑1(1:4)",
            expected_factory=lambda vocabulary: _note(
                vocabulary,
                degree=6,
                accidental=1,
                octave_offset=1,
                duration=Fraction(1, 4),
            ),
        ),
        TokenParseCase(
            name="ascii flat alias",
            text="2b↓1(1:8)",
            expected_factory=lambda vocabulary: _note(
                vocabulary,
                degree=2,
                accidental=-1,
                octave_offset=-1,
                duration=Fraction(1, 8),
            ),
        ),
        TokenParseCase(
            name="rest",
            text="r(3:8)",
            expected_factory=lambda vocabulary: _rest(vocabulary, Fraction(3, 8)),
        ),
        TokenParseCase(
            name="hold",
            text="h(1:4)",
            expected_factory=lambda vocabulary: _hold(vocabulary, Fraction(1, 4)),
        ),
        TokenParseCase(
            name="right hand",
            text="R",
            expected_factory=lambda vocabulary: HandToken(hand=Hand.RIGHT),
        ),
        TokenParseCase(
            name="left hand",
            text="L",
            expected_factory=lambda vocabulary: HandToken(hand=Hand.LEFT),
        ),
        TokenParseCase(
            name="join previous",
            text="~",
            expected_factory=lambda vocabulary: JoinWithPreviousToken(),
        ),
        TokenParseCase(
            name="bar",
            text="|",
            expected_factory=lambda vocabulary: BarToken(),
        ),
        TokenParseCase(
            name="start",
            text="BOS",
            expected_factory=lambda vocabulary: StartToken(),
        ),
        TokenParseCase(
            name="end",
            text="‖",
            expected_factory=lambda vocabulary: EndToken(),
        ),
    ]

    SEQUENCE_CASES = TestTokenTextDumping.CASES + [
        TokenSequenceCase(
            name="ascii accidental aliases normalize to canonical tokens",
            text="R 6#↑1(1:4) 2b↓1(1:8) ‖",
            expected_factory=lambda vocabulary: [
                HandToken(hand=Hand.RIGHT),
                _note(vocabulary, degree=6, accidental=1, octave_offset=1, duration=Fraction(1, 4)),
                _note(vocabulary, degree=2, accidental=-1, octave_offset=-1, duration=Fraction(1, 8)),
                EndToken(),
            ],
        ),
    ]

    @pytest.mark.parametrize("case", TOKEN_CASES, ids=str)
    def test_token_from_text_parses_single_token(
        self,
        case: TokenParseCase,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        assert token_from_text(case.text, duration_vocabulary=duration_vocabulary) == case.expected_factory(
            duration_vocabulary
        )

    @pytest.mark.parametrize("case", SEQUENCE_CASES, ids=str)
    def test_tokens_from_text_parses_sequence(
        self,
        case: TokenSequenceCase,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        assert tokens_from_text(case.text, duration_vocabulary=duration_vocabulary) == case.expected_factory(
            duration_vocabulary
        )

    def test_tokens_from_text_returns_empty_sequence_for_blank_text(
        self,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        assert tokens_from_text(" \n\t ", duration_vocabulary=duration_vocabulary) == []


class TestTokenTextErrorHandling:
    CASES = [
        TokenParseErrorCase(
            name="missing separator before end",
            text="R 1(1:4)‖",
            expected_error=TokenTextParseError,
            expected_match="position 1",
        ),
        TokenParseErrorCase(
            name="double bar is not an end token",
            text="R ||",
            expected_error=TokenTextParseError,
            expected_match="position 1",
        ),
        TokenParseErrorCase(
            name="octave offset above schema range",
            text="1↑3(1:4)",
            expected_error=TokenTextParseError,
            expected_match="octave offset",
        ),
        TokenParseErrorCase(
            name="octave offset below schema range",
            text="1↓3(1:4)",
            expected_error=TokenTextParseError,
            expected_match="octave offset",
        ),
        TokenParseErrorCase(
            name="unsupported duration",
            text="R 1(1:5)",
            expected_error=UnsupportedTokenDurationError,
            expected_match="duration 1:5",
        ),
        TokenParseErrorCase(
            name="zero duration numerator",
            text="r(0:4)",
            expected_error=TokenTextParseError,
            expected_match="duration",
        ),
        TokenParseErrorCase(
            name="zero duration denominator",
            text="r(1:0)",
            expected_error=TokenTextParseError,
            expected_match="duration",
        ),
        TokenParseErrorCase(
            name="degree below range",
            text="0(1:4)",
            expected_error=TokenTextParseError,
            expected_match="unrecognized",
        ),
        TokenParseErrorCase(
            name="degree above range",
            text="8(1:4)",
            expected_error=TokenTextParseError,
            expected_match="unrecognized",
        ),
        TokenParseErrorCase(
            name="lowercase hand is invalid",
            text="r 1(1:4)",
            expected_error=TokenTextParseError,
            expected_match="position 0",
        ),
        TokenParseErrorCase(
            name="missing octave amount",
            text="1↑(1:4)",
            expected_error=TokenTextParseError,
            expected_match="unrecognized",
        ),
        TokenParseErrorCase(
            name="slash duration syntax is invalid",
            text="1/4",
            expected_error=TokenTextParseError,
            expected_match="unrecognized",
        ),
    ]

    @pytest.mark.parametrize("case", CASES, ids=str)
    def test_tokens_from_text_raises_expected_errors(
        self,
        case: TokenParseErrorCase,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        with pytest.raises(case.expected_error, match=case.expected_match):
            tokens_from_text(case.text, duration_vocabulary=duration_vocabulary)
