"""Key-relative chord representation for synthetic generation."""

from musak_model.synthetic.harmony.expansion import (
    ChordTone,
    UnspellableChordError,
    expand_chord_to_tones,
)
from musak_model.synthetic.harmony.schema import (
    DEFAULT_CHORD_EXTENSION,
    Chord,
    ChordExtension,
    ChordQuality,
)
from musak_model.synthetic.harmony.vocabulary import (
    ChordVocabularyConfig,
    ExtensionDefinition,
    QualityDefinition,
)

__all__ = [
    "DEFAULT_CHORD_EXTENSION",
    "Chord",
    "ChordExtension",
    "ChordQuality",
    "ChordTone",
    "ChordVocabularyConfig",
    "ExtensionDefinition",
    "QualityDefinition",
    "UnspellableChordError",
    "expand_chord_to_tones",
]
