from dataclasses import dataclass

from musak_model.data.schema import Segment
from musak_model.harmony.schema import Chord
from musak_model.tokens.schema import Hand


@dataclass(frozen=True)
class BaselineSample:
    hand: Hand
    bar_index: int
    position: int
    start_in_bars: float
    register_anchor: int
    register_midi_pitch: int
    accent_weight: float


@dataclass(frozen=True)
class ChordWindowSample:
    start_in_bars: float
    end_in_bars: float
    chord: Chord


@dataclass(frozen=True)
class GenerationTrace:
    samples: tuple[BaselineSample, ...]
    grid_count_per_bar: int
    bar_count: int
    chord_windows: tuple[ChordWindowSample, ...] = ()


@dataclass(frozen=True)
class SegmentGenerationResult:
    segment: Segment
    trace: GenerationTrace
