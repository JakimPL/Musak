from __future__ import annotations

from fractions import Fraction

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedEvent, ParsedNote, ParsedRest, ParsedScore


def clean_parsed_score(score: ParsedScore) -> ParsedScore:
    return trim_silent_edge_bars(deduplicate_simultaneous_pitches(truncate_overlapping_events(score)))


def truncate_overlapping_events(score: ParsedScore) -> ParsedScore:
    return score.model_copy(
        update={
            "right_hand_bars": [_truncate_bar_overlaps(bar) for bar in score.right_hand_bars],
            "left_hand_bars": [_truncate_bar_overlaps(bar) for bar in score.left_hand_bars],
        }
    )


def deduplicate_simultaneous_pitches(score: ParsedScore) -> ParsedScore:
    return score.model_copy(
        update={
            "right_hand_bars": [_deduplicate_bar(bar) for bar in score.right_hand_bars],
            "left_hand_bars": [_deduplicate_bar(bar) for bar in score.left_hand_bars],
        }
    )


def trim_silent_edge_bars(score: ParsedScore) -> ParsedScore:
    bar_count = min(len(score.right_hand_bars), len(score.left_hand_bars))
    start = 0
    while start < bar_count and _is_silent_bar_pair(score.right_hand_bars[start], score.left_hand_bars[start]):
        start += 1

    end = bar_count
    while end > start and _is_silent_bar_pair(score.right_hand_bars[end - 1], score.left_hand_bars[end - 1]):
        end -= 1

    if start == 0 and end == bar_count:
        return score

    return score.model_copy(
        update={
            "right_hand_bars": score.right_hand_bars[start:end],
            "left_hand_bars": score.left_hand_bars[start:end],
        }
    )


def _is_silent_bar_pair(right_bar: ParsedBar, left_bar: ParsedBar) -> bool:
    return not _has_pitched_event(right_bar) and not _has_pitched_event(left_bar)


def _has_pitched_event(bar: ParsedBar) -> bool:
    return any(isinstance(event, ParsedNote | ParsedChord) for event in bar.events)


def _deduplicate_bar(bar: ParsedBar) -> ParsedBar:
    groups: dict[tuple[Fraction, Fraction], set[int]] = {}
    for event in bar.events:
        if isinstance(event, ParsedNote):
            groups.setdefault((event.beat_offset, event.duration), set()).add(event.midi_pitch)
        elif isinstance(event, ParsedChord):
            groups.setdefault((event.beat_offset, event.duration), set()).update(event.midi_pitches)

    emitted_groups: set[tuple[Fraction, Fraction]] = set()
    events: list[ParsedEvent] = []
    for event in bar.events:
        if not isinstance(event, ParsedNote | ParsedChord):
            events.append(event)
            continue

        group_key = (event.beat_offset, event.duration)
        if group_key in emitted_groups:
            continue

        emitted_groups.add(group_key)
        events.append(_pitched_event_from_group(group_key=group_key, midi_pitches=groups[group_key]))

    if events == bar.events:
        return bar

    return bar.model_copy(update={"events": events})


def _truncate_bar_overlaps(bar: ParsedBar) -> ParsedBar:
    events = sorted(bar.events, key=_event_sort_key)
    offsets = sorted({event.beat_offset for event in events})
    next_offsets = {offset: offsets[index + 1] for index, offset in enumerate(offsets[:-1])}
    truncated_events: list[ParsedEvent] = []

    for event in events:
        next_offset = next_offsets.get(event.beat_offset)
        if next_offset is None or event.beat_offset + event.duration <= next_offset:
            truncated_events.append(event)
            continue

        duration = next_offset - event.beat_offset
        if duration > 0:
            truncated_events.append(_with_duration(event, duration))

    if truncated_events == bar.events:
        return bar

    return bar.model_copy(update={"events": truncated_events})


def _with_duration(event: ParsedEvent, duration: Fraction) -> ParsedEvent:
    if isinstance(event, ParsedNote | ParsedRest | ParsedChord):
        return event.model_copy(update={"duration": duration})

    raise TypeError(f"unsupported parsed event: {type(event).__name__}")


def _event_sort_key(event: ParsedEvent) -> tuple[Fraction, int]:
    return event.beat_offset, _lowest_pitch(event)


def _lowest_pitch(event: ParsedEvent) -> int:
    if isinstance(event, ParsedNote):
        return event.midi_pitch

    if isinstance(event, ParsedChord):
        return min(event.midi_pitches)

    return -1


def _pitched_event_from_group(
    *, group_key: tuple[Fraction, Fraction], midi_pitches: set[int]
) -> ParsedNote | ParsedChord:
    beat_offset, duration = group_key
    unique_pitches = sorted(midi_pitches)
    if not unique_pitches:
        raise ValueError("cannot build a pitched event without pitches")

    if len(unique_pitches) == 1:
        return ParsedNote(midi_pitch=unique_pitches[0], duration=duration, beat_offset=beat_offset)

    return ParsedChord(midi_pitches=unique_pitches, duration=duration, beat_offset=beat_offset)
