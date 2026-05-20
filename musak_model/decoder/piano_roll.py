from fractions import Fraction

from pydantic import BaseModel, ConfigDict, Field

from musak_model.data.schema import ParsedChord, ParsedNote, ParsedScore, Segment, SegmentMetadata
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_token_to_midi_pitch
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    StartToken,
    Token,
)
from musak_shared.elements import MIDI_MAX_PITCH


class PianoRollEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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

    return sorted(
        events,
        key=lambda event: (
            event.start,
            event.hand.value,
            event.midi_pitch,
        ),
    )


def segment_to_piano_roll_events(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[PianoRollEvent]:
    return tokens_to_piano_roll_events(
        segment.tokens,
        metadata=segment.metadata,
        duration_vocabulary=duration_vocabulary,
        default_hand=Hand.RIGHT,
    )


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
    cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
    last_attack_indices: dict[Hand, list[int]] = {Hand.RIGHT: [], Hand.LEFT: []}
    events: list[PianoRollEvent] = []

    for token_index, token in enumerate(tokens):
        if isinstance(token, HandToken):
            active_hand = token.hand
            continue

        if isinstance(token, StartToken):
            continue

        if isinstance(token, BarToken):
            bar_index += 1
            cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
            continue

        if isinstance(token, EndToken):
            break

        if isinstance(token, RestToken):
            cursors[active_hand] += duration_vocabulary.id_to_fraction(token.duration_id)
            continue

        if isinstance(token, HoldToken):
            duration = duration_vocabulary.id_to_fraction(token.duration_id)
            _extend_last_attack(
                events,
                event_indices=last_attack_indices[active_hand],
                duration=duration,
                hand=active_hand,
            )
            cursors[active_hand] += duration
            continue

        if isinstance(token, NoteToken):
            duration = duration_vocabulary.id_to_fraction(token.duration_id)
            event = PianoRollEvent(
                hand=active_hand,
                midi_pitch=note_token_to_midi_pitch(
                    token,
                    scale_root=metadata.scale_root,
                    scale_type=metadata.scale_type,
                    hand=active_hand,
                ),
                start=bar_index * measure_duration + cursors[active_hand],
                duration=duration,
                token_index=token_index,
                token_text=token.to_text(duration_vocabulary=duration_vocabulary),
            )
            events.append(event)
            cursors[active_hand] += duration
            last_attack_indices[active_hand] = [len(events) - 1]
            continue

        if isinstance(token, JoinWithPreviousToken):
            if len(events) < 2:
                raise ValueError("join-with-previous token needs at least two decoded notes")

            previous_event = events[-1]
            joined_start = events[-2].start
            joined_event = previous_event.model_copy(update={"start": joined_start})
            events[-1] = joined_event
            last_attack_indices[active_hand] = _same_onset_event_indices(
                events,
                hand=active_hand,
                start=joined_start,
            )
            cursors[active_hand] = max(
                cursors[active_hand] - joined_event.duration,
                joined_event.end - bar_index * measure_duration,
            )
            continue

    return events


def _extend_last_attack(
    events: list[PianoRollEvent],
    *,
    event_indices: list[int],
    duration: Fraction,
    hand: Hand,
) -> None:
    if not event_indices:
        raise ValueError(f"hold token needs a previous {hand.value} hand note or chord")

    for event_index in event_indices:
        event = events[event_index]
        if event.hand != hand:
            raise ValueError("hold token cannot extend a note from another hand")

        events[event_index] = event.model_copy(update={"duration": event.duration + duration})


def _same_onset_event_indices(events: list[PianoRollEvent], *, hand: Hand, start: Fraction) -> list[int]:
    return [index for index, event in enumerate(events) if event.hand == hand and event.start == start]
