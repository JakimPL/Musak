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


def test_default_config_defines_supported_extensions() -> None:
    config = ChordVocabularyConfig.load()

    assert config.extension_definition(ChordExtension.TRIAD).additional_intervals == ()
    assert config.extension_definition(ChordExtension.SEVENTH).additional_intervals == (10,)
    assert config.extension_definition(ChordExtension.MAJOR_SEVENTH).additional_intervals == (11,)


def test_quality_definition_rejects_non_triad_intervals() -> None:
    with pytest.raises(ValidationError, match="triad"):
        QualityDefinition(intervals=(0, 4, 7, 10), enabled=True)


def test_quality_definition_rejects_non_zero_root() -> None:
    with pytest.raises(ValidationError, match="root offset"):
        QualityDefinition(intervals=(1, 4, 7), enabled=True)


def test_extension_definition_rejects_out_of_range_interval() -> None:
    with pytest.raises(ValidationError, match="extension intervals"):
        ExtensionDefinition(additional_intervals=(12,), enabled=False)
