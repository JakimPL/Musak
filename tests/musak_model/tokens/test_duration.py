from dataclasses import dataclass
from fractions import Fraction

import pytest

from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary


@dataclass(frozen=True)
class VocabularyGenerationCase:
    name: str
    config: TokenizationConfig
    expected_fractions: tuple[Fraction, ...]

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ClosestDurationCase:
    name: str
    target: Fraction
    expected: Fraction

    def __str__(self) -> str:
        return self.name


class TestDurationVocabularyGeneration:
    CASES = [
        VocabularyGenerationCase(
            name="base durations without tuplets",
            config=TokenizationConfig(shortest_duration=4, allowed_tuplets=(), max_dots=0),
            expected_fractions=(
                Fraction(1, 1),
                Fraction(1, 2),
                Fraction(1, 4),
            ),
        ),
        VocabularyGenerationCase(
            name="base and triplet durations",
            config=TokenizationConfig(shortest_duration=4, allowed_tuplets=(3,), max_dots=0),
            expected_fractions=(
                Fraction(1, 1),
                Fraction(1, 2),
                Fraction(1, 3),
                Fraction(1, 4),
                Fraction(1, 6),
                Fraction(1, 12),
            ),
        ),
        VocabularyGenerationCase(
            name="single dot expansion",
            config=TokenizationConfig(shortest_duration=4, allowed_tuplets=(3,), max_dots=1),
            expected_fractions=(
                Fraction(1, 1),
                Fraction(3, 4),
                Fraction(1, 2),
                Fraction(3, 8),
                Fraction(1, 3),
                Fraction(1, 4),
                Fraction(1, 6),
                Fraction(1, 8),
                Fraction(1, 12),
            ),
        ),
        VocabularyGenerationCase(
            name="double dot expansion",
            config=TokenizationConfig(shortest_duration=4, allowed_tuplets=(3,), max_dots=2),
            expected_fractions=(
                Fraction(1, 1),
                Fraction(7, 8),
                Fraction(3, 4),
                Fraction(7, 12),
                Fraction(1, 2),
                Fraction(7, 16),
                Fraction(3, 8),
                Fraction(1, 3),
                Fraction(7, 24),
                Fraction(1, 4),
                Fraction(1, 6),
                Fraction(7, 48),
                Fraction(1, 8),
                Fraction(1, 12),
            ),
        ),
        VocabularyGenerationCase(
            name="multiple tuplet divisors",
            config=TokenizationConfig(shortest_duration=8, allowed_tuplets=(3, 5), max_dots=0),
            expected_fractions=(
                Fraction(1, 1),
                Fraction(1, 2),
                Fraction(1, 3),
                Fraction(1, 4),
                Fraction(1, 5),
                Fraction(1, 6),
                Fraction(1, 8),
                Fraction(1, 10),
                Fraction(1, 12),
                Fraction(1, 20),
                Fraction(1, 24),
                Fraction(1, 40),
            ),
        ),
    ]

    @pytest.mark.parametrize("case", CASES, ids=str)
    def test_generates_exact_sorted_fraction_vocabulary(self, case: VocabularyGenerationCase) -> None:
        vocabulary = DurationVocabulary(case.config)

        assert vocabulary.all_fractions() == case.expected_fractions
        assert vocabulary.vocabulary_size() == len(case.expected_fractions)

    @pytest.mark.parametrize("case", CASES, ids=str)
    def test_fraction_ids_follow_sorted_vocabulary_order(self, case: VocabularyGenerationCase) -> None:
        vocabulary = DurationVocabulary(case.config)

        assert [vocabulary.fraction_to_id(duration) for duration in case.expected_fractions] == list(
            range(len(case.expected_fractions))
        )


class TestDurationVocabularyLookup:
    @pytest.fixture
    def vocabulary(self) -> DurationVocabulary:
        return DurationVocabulary(TokenizationConfig(shortest_duration=4, allowed_tuplets=(3,), max_dots=0))

    def test_fraction_to_id_and_id_to_fraction_are_inverses(self, vocabulary: DurationVocabulary) -> None:
        for duration in vocabulary.all_fractions():
            duration_id = vocabulary.fraction_to_id(duration)

            assert vocabulary.id_to_fraction(duration_id) == duration

    def test_fraction_to_id_rejects_unsupported_fraction(self, vocabulary: DurationVocabulary) -> None:
        with pytest.raises(KeyError):
            vocabulary.fraction_to_id(Fraction(1, 5))


class TestDurationVocabularyClosestMatch:
    CASES = [
        ClosestDurationCase(name="exact match", target=Fraction(1, 4), expected=Fraction(1, 4)),
        ClosestDurationCase(name="between quarter and sixth", target=Fraction(1, 5), expected=Fraction(1, 6)),
        ClosestDurationCase(name="above largest duration", target=Fraction(5, 4), expected=Fraction(1, 1)),
        ClosestDurationCase(name="below smallest duration", target=Fraction(1, 20), expected=Fraction(1, 12)),
        ClosestDurationCase(
            name="tie chooses earlier sorted duration", target=Fraction(5, 12), expected=Fraction(1, 2)
        ),
    ]

    @pytest.fixture
    def vocabulary(self) -> DurationVocabulary:
        return DurationVocabulary(TokenizationConfig(shortest_duration=4, allowed_tuplets=(3,), max_dots=0))

    @pytest.mark.parametrize("case", CASES, ids=str)
    def test_find_closest_returns_id_and_fraction(
        self,
        case: ClosestDurationCase,
        vocabulary: DurationVocabulary,
    ) -> None:
        duration_id, duration = vocabulary.find_closest(case.target)

        assert duration == case.expected
        assert vocabulary.id_to_fraction(duration_id) == duration
