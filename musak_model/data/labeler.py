from fractions import Fraction

from musak_model.data.converter import pitch_to_degree
from musak_model.data.schema import (
    DifficultyFeatures,
    ParsedBar,
    ParsedChord,
    ParsedEvent,
    ParsedNote,
    ParsedScore,
    Segment,
)
from musak_model.tokens.schema import (
    DURATION_FRACTIONS,
    DurationClass,
    Hand,
    NoteToken,
    ScaleType,
    Token,
)

_DOTTED_DURATION_CLASSES: frozenset[DurationClass] = frozenset(
    {
        DurationClass.DOTTED_HALF,
        DurationClass.DOTTED_QUARTER,
        DurationClass.DOTTED_EIGHTH,
    }
)


def extract_difficulty_features(segment: Segment, *, score: ParsedScore, scale_type: ScaleType) -> DifficultyFeatures:
    window_bars_right = _select_window_bars(score.right_hand_bars, bar_count=segment.bar_count)
    window_bars_left = _select_window_bars(score.left_hand_bars, bar_count=segment.bar_count)

    return DifficultyFeatures(
        max_right_hand_span_semitones=_max_hand_span(window_bars_right),
        max_left_hand_span_semitones=_max_hand_span(window_bars_left),
        notes_per_beat=_notes_per_beat(window_bars_right + window_bars_left, score=score),
        rhythmic_diversity=_rhythmic_diversity(segment),
        voice_independence=_voice_independence(segment),
        has_accidentals=_has_accidentals(window_bars_right + window_bars_left, score=score, scale_type=scale_type),
        has_dotted_notes=_has_dotted_notes(segment),
    )


def _select_window_bars(bars: list[ParsedBar], *, bar_count: int) -> list[ParsedBar]:
    return bars[:bar_count]


def _max_hand_span(bars: list[ParsedBar]) -> int:
    pitches_per_bar = [_collect_pitches(bar) for bar in bars]
    spans = [max(pitches) - min(pitches) for pitches in pitches_per_bar if len(pitches) >= 2]
    return max(spans) if spans else 0


def _collect_pitches(bar: ParsedBar) -> list[int]:
    pitches: list[int] = []
    for event in bar.events:
        if isinstance(event, ParsedNote):
            pitches.append(event.midi_pitch)
        elif isinstance(event, ParsedChord):
            pitches.extend(event.midi_pitches)

    return pitches


def _notes_per_beat(bars: list[ParsedBar], *, score: ParsedScore) -> float:
    beat_duration = Fraction(1, score.time_denominator)
    total_beats = len(bars) * Fraction(score.time_numerator, score.time_denominator) / beat_duration
    note_count = sum(1 for bar in bars for event in bar.events if isinstance(event, (ParsedNote, ParsedChord)))

    return float(note_count / total_beats) if total_beats > 0 else 0.0


def _rhythmic_diversity(segment: Segment) -> float:
    all_tokens = segment.right_hand_tokens + segment.left_hand_tokens
    durations_present: set[DurationClass] = set()
    for token in all_tokens:
        if isinstance(token, NoteToken):
            durations_present.add(token.duration)

    return len(durations_present) / len(DurationClass)


def _voice_independence(segment: Segment) -> float:
    right_rhythm = _rhythm_vector(segment.right_hand_tokens)
    left_rhythm = _rhythm_vector(segment.left_hand_tokens)

    if len(right_rhythm) != len(left_rhythm) or not right_rhythm:
        return 0.0

    matching = sum(1 for rh, lh in zip(right_rhythm, left_rhythm) if rh == lh)
    return 1.0 - matching / len(right_rhythm)


def _rhythm_vector(tokens: list[Token]) -> list[Fraction]:
    rhythm: list[Fraction] = []
    for token in tokens:
        if isinstance(token, NoteToken):
            rhythm.append(DURATION_FRACTIONS[token.duration])

    return rhythm


def _has_accidentals(bars: list[ParsedBar], *, score: ParsedScore, scale_type: ScaleType) -> bool:
    for bar in bars:
        for event in bar.events:
            if isinstance(event, ParsedNote):
                pitch_degree = pitch_to_degree(
                    event.midi_pitch,
                    key_root=score.key_root,
                    scale_type=scale_type,
                    hand=Hand.RIGHT,
                )
                if pitch_degree.accidental != 0:
                    return True

            elif isinstance(event, ParsedChord):
                for midi_pitch in event.midi_pitches:
                    pitch_degree = pitch_to_degree(
                        midi_pitch,
                        key_root=score.key_root,
                        scale_type=scale_type,
                        hand=Hand.RIGHT,
                    )
                    if pitch_degree.accidental != 0:
                        return True
    return False


def _has_dotted_notes(segment: Segment) -> bool:
    for token in segment.right_hand_tokens + segment.left_hand_tokens:
        if isinstance(token, NoteToken) and token.duration in _DOTTED_DURATION_CLASSES:
            return True

    return False


def _note_events(bar: ParsedBar) -> list[ParsedEvent]:
    return [event for event in bar.events if isinstance(event, (ParsedNote, ParsedChord))]
