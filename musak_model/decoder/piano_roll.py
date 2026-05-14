from fractions import Fraction

from pydantic import BaseModel, ConfigDict, Field

from musak_model.common.elements import MIDI_MAX_PITCH, MIDI_OCTAVE_OFFSET, PITCHES_PER_OCTAVE
from musak_model.data.schema import ParsedChord, ParsedNote, ParsedScore, Segment, SegmentMetadata
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    HAND_HOME_OCTAVES,
    SCALE_INTERVALS,
    BarToken,
    EndToken,
    Hand,
    HandToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    Token,
)


class PianoRollEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    hand: Hand
    midi_pitch: int = Field(ge=0, le=MIDI_MAX_PITCH)
    start: Fraction = Field(ge=0)
    duration: Fraction = Field(gt=0)
    token_index: int | None = None
    token_text: str | None = None

    @property
    def end(self) -> Fraction:
        return self.start + self.duration


def parsed_score_to_piano_roll_events(score: ParsedScore) -> list[PianoRollEvent]:
    measure_duration = Fraction(score.time_numerator, score.time_denominator)
    events: list[PianoRollEvent] = []

    for hand, bars in ((Hand.RIGHT, score.right_hand_bars), (Hand.LEFT, score.left_hand_bars)):
        for bar_index, bar in enumerate(bars):
            bar_start = bar_index * measure_duration
            for parsed_event in bar.events:
                start = bar_start + parsed_event.beat_offset
                if isinstance(parsed_event, ParsedNote):
                    events.append(
                        PianoRollEvent(
                            hand=hand,
                            midi_pitch=parsed_event.midi_pitch,
                            start=start,
                            duration=parsed_event.duration,
                        )
                    )
                elif isinstance(parsed_event, ParsedChord):
                    events.extend(
                        PianoRollEvent(
                            hand=hand,
                            midi_pitch=midi_pitch,
                            start=start,
                            duration=parsed_event.duration,
                        )
                        for midi_pitch in parsed_event.midi_pitches
                    )

    return sorted(events, key=lambda event: (event.start, event.hand.value, event.midi_pitch))


def segment_to_piano_roll_events(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[PianoRollEvent]:
    if segment.tokens:
        return tokens_to_piano_roll_events(
            segment.tokens,
            metadata=segment.metadata,
            duration_vocabulary=duration_vocabulary,
            default_hand=Hand.RIGHT,
        )

    right_events = tokens_to_piano_roll_events(
        segment.right_hand_tokens,
        metadata=segment.metadata,
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )
    left_events = tokens_to_piano_roll_events(
        segment.left_hand_tokens,
        metadata=segment.metadata,
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.LEFT,
    )
    return sorted(right_events + left_events, key=lambda event: (event.start, event.hand.value, event.midi_pitch))


def tokens_to_piano_roll_events(
    tokens: list[Token],
    *,
    metadata: SegmentMetadata,
    duration_vocabulary: DurationVocabulary,
    default_hand: Hand,
) -> list[PianoRollEvent]:
    measure_duration = Fraction(metadata.time_numerator, metadata.time_denominator)
    active_hand = default_hand
    bar_index = 0
    cursor = Fraction(0)
    events: list[PianoRollEvent] = []

    for token_index, token in enumerate(tokens):
        if isinstance(token, HandToken):
            active_hand = token.hand
            continue

        if isinstance(token, BarToken):
            bar_index += 1
            cursor = Fraction(0)
            continue

        if isinstance(token, EndToken):
            break

        if isinstance(token, RestToken):
            cursor += duration_vocabulary.id_to_fraction(token.duration_id)
            continue

        if isinstance(token, NoteToken):
            duration = duration_vocabulary.id_to_fraction(token.duration_id)
            event = PianoRollEvent(
                hand=active_hand,
                midi_pitch=_token_to_midi_pitch(token, metadata=metadata, hand=active_hand),
                start=bar_index * measure_duration + cursor,
                duration=duration,
                token_index=token_index,
                token_text=token.to_text(duration_vocabulary=duration_vocabulary),
            )
            events.append(event)
            cursor += duration
            continue

        if isinstance(token, JoinWithPreviousToken):
            if len(events) < 2:
                raise ValueError("join-with-previous token needs at least two decoded notes")

            previous_event = events[-1]
            joined_start = events[-2].start
            joined_event = previous_event.model_copy(update={"start": joined_start})
            events[-1] = joined_event
            cursor = max(cursor - joined_event.duration, joined_event.end - bar_index * measure_duration)
            continue

    return events


def _token_to_midi_pitch(token: NoteToken, *, metadata: SegmentMetadata, hand: Hand) -> int:
    interval = SCALE_INTERVALS[metadata.scale_type][token.degree - 1]
    pitch_class = (metadata.key_root + interval + token.accidental) % PITCHES_PER_OCTAVE
    octave = HAND_HOME_OCTAVES[hand] + token.octave_offset
    return (octave + MIDI_OCTAVE_OFFSET) * PITCHES_PER_OCTAVE + pitch_class
