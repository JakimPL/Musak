from fractions import Fraction

from musak_model.data.converter import pitch_to_degree
from musak_model.data.schema import (
    DifficultyFeatures,
    ParsedBar,
    ParsedChord,
    ParsedNote,
    ParsedScore,
    Segment,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    Hand,
    NoteToken,
    ScaleType,
    Token,
)
from musak_model.tokens.views import tokens_for_hand
from musak_shared.elements import DOTTED_LIKE_DURATIONS


def extract_difficulty_features(
    segment: Segment,
    *,
    score: ParsedScore,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> DifficultyFeatures:
    window_bars_right = _select_window_bars(
        score.right_hand_bars,
        start=segment.metadata.window_start_bar,
        bar_count=segment.bar_count,
    )
    window_bars_left = _select_window_bars(
        score.left_hand_bars,
        start=segment.metadata.window_start_bar,
        bar_count=segment.bar_count,
    )

    return DifficultyFeatures(
        max_right_hand_span_semitones=_max_hand_span(window_bars_right),
        max_left_hand_span_semitones=_max_hand_span(window_bars_left),
        notes_per_beat=_notes_per_beat(window_bars_right + window_bars_left),
        rhythmic_diversity=_rhythmic_diversity(segment, duration_vocabulary=duration_vocabulary),
        voice_independence=_voice_independence(segment, duration_vocabulary=duration_vocabulary),
        has_accidentals=(
            _has_accidentals(window_bars_right, segment=segment, hand=Hand.RIGHT)
            or _has_accidentals(window_bars_left, segment=segment, hand=Hand.LEFT)
        ),
        has_dotted_notes=_has_dotted_notes(segment, duration_vocabulary=duration_vocabulary),
    )


def _select_window_bars(
    bars: list[ParsedBar],
    *,
    start: int,
    bar_count: int,
) -> list[ParsedBar]:
    return bars[start : start + bar_count]


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


def _notes_per_beat(
    bars: list[ParsedBar],
) -> float:
    total_beats = sum(bar.time_numerator for bar in bars)
    note_count = sum(1 for bar in bars for event in bar.events if isinstance(event, (ParsedNote, ParsedChord)))
    return float(note_count / total_beats) if total_beats > 0 else 0.0


def _rhythmic_diversity(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> float:
    durations_present: set[int] = set()
    for token in segment.tokens:
        if isinstance(token, NoteToken):
            durations_present.add(token.duration_id)

    return len(durations_present) / duration_vocabulary.vocabulary_size()


def _voice_independence(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> float:
    right_rhythm = _rhythm_vector(
        tokens_for_hand(segment.tokens, hand=Hand.RIGHT, include_structure=False),
        duration_vocabulary=duration_vocabulary,
    )
    left_rhythm = _rhythm_vector(
        tokens_for_hand(segment.tokens, hand=Hand.LEFT, include_structure=False),
        duration_vocabulary=duration_vocabulary,
    )

    if len(right_rhythm) != len(left_rhythm) or not right_rhythm:
        return 0.0

    matching = sum(1 for rh, lh in zip(right_rhythm, left_rhythm) if rh == lh)
    return 1.0 - matching / len(right_rhythm)


def _rhythm_vector(
    tokens: list[Token],
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[Fraction]:
    rhythm: list[Fraction] = []
    for token in tokens:
        if isinstance(token, NoteToken):
            rhythm.append(duration_vocabulary.id_to_fraction(token.duration_id))

    return rhythm


def _has_accidentals(
    bars: list[ParsedBar],
    *,
    segment: Segment,
    hand: Hand,
) -> bool:
    key_fifths = _segment_key_fifths(segment)
    for bar in bars:
        for event in bar.events:
            if isinstance(event, ParsedNote):
                pitch_degree = pitch_to_degree(
                    event.midi_pitch,
                    scale_root=segment.scale_root,
                    key_fifths=key_fifths,
                    scale_type=segment.scale_type,
                    hand=hand,
                )
                if pitch_degree.accidental != 0:
                    return True

            elif isinstance(event, ParsedChord):
                for midi_pitch in event.midi_pitches:
                    pitch_degree = pitch_to_degree(
                        midi_pitch,
                        scale_root=segment.scale_root,
                        key_fifths=key_fifths,
                        scale_type=segment.scale_type,
                        hand=hand,
                    )
                    if pitch_degree.accidental != 0:
                        return True

    return False


def _segment_key_fifths(segment: Segment) -> int:
    if segment.metadata.scale_match is not None and segment.metadata.scale_match.declared_key_fifths is not None:
        return segment.metadata.scale_match.declared_key_fifths

    return 0


def _has_dotted_notes(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> bool:
    return _has_note_duration_in(
        segment,
        duration_vocabulary=duration_vocabulary,
        durations=DOTTED_LIKE_DURATIONS,
    )


def _has_note_duration_in(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    durations: frozenset[Fraction],
) -> bool:
    for token in segment.tokens:
        if isinstance(token, NoteToken) and duration_vocabulary.id_to_fraction(token.duration_id) in durations:
            return True

    return False
