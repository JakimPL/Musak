from fractions import Fraction

from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration_vocabulary import DurationVocabulary


def test_duration_vocabulary_basic_generation() -> None:
    """Test that vocabulary generates base durations correctly."""
    config = TokenizationConfig(
        shortest_duration=16,
        max_tuplets=(3,),
        max_dots=1,
    )
    vocab = DurationVocabulary(config)

    fractions = vocab.all_fractions()

    # Should include whole, half, quarter, eighth, sixteenth (base durations)
    assert Fraction(1, 1) in fractions  # Whole
    assert Fraction(1, 2) in fractions  # Half
    assert Fraction(1, 4) in fractions  # Quarter
    assert Fraction(1, 8) in fractions  # Eighth
    assert Fraction(1, 16) in fractions  # Sixteenth


def test_duration_vocabulary_tuplets() -> None:
    """Test that vocabulary generates tuplet durations."""
    config = TokenizationConfig(
        shortest_duration=32,  # Allow shorter tuplets
        max_tuplets=(3, 5),
        max_dots=1,
    )
    vocab = DurationVocabulary(config)

    fractions = vocab.all_fractions()

    # Triplet eighth: 1/8 / 3 = 1/24
    assert Fraction(1, 24) in fractions
    # Triplet quarter: 1/4 / 3 = 1/12
    assert Fraction(1, 12) in fractions
    # Quintuplet sixteenth: 1/16 / 5 = 1/80
    assert Fraction(1, 80) in fractions


def test_duration_vocabulary_dotted() -> None:
    """Test that vocabulary generates dotted durations."""
    config = TokenizationConfig(
        shortest_duration=16,
        max_tuplets=(3,),
        max_dots=1,
    )
    vocab = DurationVocabulary(config)

    fractions = vocab.all_fractions()

    # Dotted quarter: 1/4 * 1.5 = 3/8
    assert Fraction(3, 8) in fractions
    # Dotted eighth: 1/8 * 1.5 = 3/16
    assert Fraction(3, 16) in fractions


def test_duration_vocabulary_double_dot() -> None:
    """Test that vocabulary generates double-dotted durations when max_dots=2."""
    config = TokenizationConfig(
        shortest_duration=16,
        max_tuplets=(3,),
        max_dots=2,
    )
    vocab = DurationVocabulary(config)

    fractions = vocab.all_fractions()

    # Double-dotted quarter: 1/4 * 1.75 = 7/16
    assert Fraction(7, 16) in fractions


def test_duration_vocabulary_find_closest() -> None:
    """Test that find_closest maps durations to nearest in vocabulary."""
    config = TokenizationConfig(
        shortest_duration=16,
        max_tuplets=(3,),
        max_dots=1,
    )
    vocab = DurationVocabulary(config)

    # Exact match: quarter note
    duration_id, closest = vocab.find_closest(Fraction(1, 4))
    assert closest == Fraction(1, 4)

    # Off by a bit: should find nearest
    duration_id, closest = vocab.find_closest(Fraction(1, 5))
    assert closest in vocab.all_fractions()


def test_duration_vocabulary_fraction_to_id_roundtrip() -> None:
    """Test that fraction_to_id and id_to_fraction are inverses."""
    config = TokenizationConfig(
        shortest_duration=16,
        max_tuplets=(3,),
        max_dots=1,
    )
    vocab = DurationVocabulary(config)

    for frac in vocab.all_fractions():
        duration_id = vocab.fraction_to_id(frac)
        recovered_frac = vocab.id_to_fraction(duration_id)
        assert recovered_frac == frac


def test_duration_vocabulary_size() -> None:
    """Test that vocabulary size is reasonable."""
    config = TokenizationConfig(
        shortest_duration=16,
        max_tuplets=(3, 5),
        max_dots=2,
    )
    vocab = DurationVocabulary(config)

    # With tuplets and double dots, should have many durations
    assert vocab.vocab_size() > 20
