import pytest
from pydantic import ValidationError

from musak_model.harmony.schema import ChordExtension, ChordQuality
from musak_model.harmony.vocabulary import (
    ChordVocabularyConfig,
    ExtensionDefinition,
    QualityDefinition,
)


def test_default_config_loads() -> None:
    config = ChordVocabularyConfig.load()

    assert set(config.enabled_qualities()) == {
        ChordQuality.MAJOR,
        ChordQuality.MINOR,
        ChordQuality.DIMINISHED,
        ChordQuality.AUGMENTED,
    }
    assert config.enabled_extensions() == (ChordExtension.TRIAD,)


def test_default_config_defines_disabled_extensions() -> None:
    config = ChordVocabularyConfig.load()

    assert config.extension_definition(ChordExtension.SEVENTH).members == 4
    assert config.extension_definition(ChordExtension.FLAT_NINTH).alterations == {4: -1}


def test_quality_definition_rejects_non_triad_intervals() -> None:
    with pytest.raises(ValidationError, match="triad"):
        QualityDefinition(intervals=(0, 4, 7, 10), enabled=True)


def test_quality_definition_rejects_non_zero_root() -> None:
    with pytest.raises(ValidationError, match="root offset"):
        QualityDefinition(intervals=(1, 4, 7), enabled=True)


def test_extension_definition_rejects_negative_alteration_member() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        ExtensionDefinition(members=5, alterations={-1: 1}, enabled=False)
