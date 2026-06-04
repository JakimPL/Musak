from __future__ import annotations

from musak_model.conditioning.harmony.planner import annotate_harmonic_plan_windows
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow, harmonic_plan_windows_from_chord_windows
from musak_model.data.schema import Segment
from musak_model.harmony.decoding.schema import ChordDecoder
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.duration import DurationVocabulary


def harmonic_plan_windows_from_segment(
    segment: Segment,
    *,
    decoder: ChordDecoder,
    duration_vocabulary: DurationVocabulary,
    vocabulary: ChordVocabularyConfig,
) -> tuple[HarmonicPlanWindow, ...]:
    return annotate_harmonic_plan_windows(
        harmonic_plan_windows_from_chord_windows(
            decoder.decode(
                segment,
                duration_vocabulary=duration_vocabulary,
                vocabulary=vocabulary,
            )
        )
    )
