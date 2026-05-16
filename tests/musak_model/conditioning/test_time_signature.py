import pytest

from musak_model.conditioning.time_signature import TimeSignatureVocabulary, TimeSignatureVocabularyConfig


def test_time_signature_vocabulary_generates_deterministic_mapping() -> None:
    vocabulary = TimeSignatureVocabulary(TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2))

    assert vocabulary.all_time_signatures() == (
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        (5, 4),
        (6, 4),
        (7, 4),
    )
    assert vocabulary.vocabulary_size == 11
    assert vocabulary.time_signature_to_id((4, 4)) == 7
    assert vocabulary.id_to_time_signature(7) == (4, 4)


def test_time_signature_vocabulary_rejects_values_outside_generated_range() -> None:
    vocabulary = TimeSignatureVocabulary(TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2))

    with pytest.raises(ValueError, match="unsupported time signature"):
        vocabulary.time_signature_to_id((8, 4))

    with pytest.raises(ValueError, match="unsupported time signature"):
        vocabulary.time_signature_to_id((1, 8))


def test_time_signature_vocabulary_config_requires_power_of_two_denominator() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        TimeSignatureVocabularyConfig(max_denominator=12, relative_numerator_range=2)


def test_time_signature_vocabulary_config_has_no_defaults() -> None:
    with pytest.raises(ValueError, match="max_denominator"):
        TimeSignatureVocabularyConfig.model_validate({"relative_numerator_range": 2})

    with pytest.raises(ValueError, match="relative_numerator_range"):
        TimeSignatureVocabularyConfig.model_validate({"max_denominator": 4})
