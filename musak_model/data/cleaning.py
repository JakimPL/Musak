from fractions import Fraction

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedEvent, ParsedNote, ParsedRest, ParsedScore


def clean_parsed_score(score: ParsedScore) -> ParsedScore:
    return trim_silent_edge_bars(
        deduplicate_simultaneous_pitches(
            truncate_overlapping_events(normalize_simultaneous_event_durations(remove_rests_overlapping_pitches(score)))
        )
    )


def remove_rests_overlapping_pitches(score: ParsedScore) -> ParsedScore:
    return score.model_copy(
        update={
            "right_hand_bars": [_remove_bar_rests_overlapping_pitches(bar) for bar in score.right_hand_bars],
            "left_hand_bars": [_remove_bar_rests_overlapping_pitches(bar) for bar in score.left_hand_bars],
        }
    )


def normalize_simultaneous_event_durations(score: ParsedScore) -> ParsedScore:
    return score.model_copy(
        update={
            "right_hand_bars": [_normalize_bar_simultaneous_durations(bar) for bar in score.right_hand_bars],
            "left_hand_bars": [_normalize_bar_simultaneous_durations(bar) for bar in score.left_hand_bars],
        }
    )


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
    while start < bar_count and is_silent_bar_pair(score.right_hand_bars[start], score.left_hand_bars[start]):
        start += 1

    end = bar_count
    while end > start and is_silent_bar_pair(score.right_hand_bars[end - 1], score.left_hand_bars[end - 1]):
        end -= 1

    if start == 0 and end == bar_count:
        return score

    return score.model_copy(
        update={
            "right_hand_bars": score.right_hand_bars[start:end],
            "left_hand_bars": score.left_hand_bars[start:end],
        }
    )


def is_silent_bar_pair(right_bar: ParsedBar, left_bar: ParsedBar) -> bool:
    return not _has_pitched_event(right_bar) and not _has_pitched_event(left_bar)


def _has_pitched_event(bar: ParsedBar) -> bool:
    return any(isinstance(event, ParsedNote | ParsedChord) for event in bar.events)


def _remove_bar_rests_overlapping_pitches(bar: ParsedBar) -> ParsedBar:
    pitched_intervals = [
        (event.beat_offset, event.beat_offset + event.duration)
        for event in bar.events
        if isinstance(event, ParsedNote | ParsedChord)
    ]
    if not pitched_intervals:
        return bar

    events = [
        event
        for event in bar.events
        if not isinstance(event, ParsedRest) or not _overlaps_any_interval(event, pitched_intervals)
    ]
    if events == bar.events:
        return bar

    return bar.model_copy(update={"events": events})


def _overlaps_any_interval(event: ParsedRest, intervals: list[tuple[Fraction, Fraction]]) -> bool:
    event_start = event.beat_offset
    event_end = event.beat_offset + event.duration
    return any(event_start < interval_end and event_end > interval_start for interval_start, interval_end in intervals)


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


def _normalize_bar_simultaneous_durations(bar: ParsedBar) -> ParsedBar:
    events = sorted(bar.events, key=_event_sort_key)
    next_offsets = _next_offsets(events)
    pitched_events_by_offset: dict[Fraction, list[ParsedNote | ParsedChord]] = {}
    for event in events:
        if isinstance(event, ParsedNote | ParsedChord):
            pitched_events_by_offset.setdefault(event.beat_offset, []).append(event)

    normalized_events: list[ParsedEvent] = []
    for event in events:
        if not isinstance(event, ParsedNote | ParsedChord):
            normalized_events.append(event)
            continue

        simultaneous_events = pitched_events_by_offset[event.beat_offset]
        durations = {simultaneous_event.duration for simultaneous_event in simultaneous_events}
        if len(simultaneous_events) < 2 or len(durations) == 1:
            normalized_events.append(event)
            continue

        normalized_duration = _shared_simultaneous_duration(
            beat_offset=event.beat_offset,
            durations=durations,
            next_offsets=next_offsets,
        )
        normalized_events.append(_with_duration(event, normalized_duration))

    if normalized_events == bar.events:
        return bar

    return bar.model_copy(update={"events": normalized_events})


def _next_offsets(events: list[ParsedEvent]) -> dict[Fraction, Fraction]:
    offsets = sorted({event.beat_offset for event in events})
    return {offset: offsets[index + 1] for index, offset in enumerate(offsets[:-1])}


def _shared_simultaneous_duration(
    *,
    beat_offset: Fraction,
    durations: set[Fraction],
    next_offsets: dict[Fraction, Fraction],
) -> Fraction:
    next_offset = next_offsets.get(beat_offset)
    if next_offset is not None:
        return next_offset - beat_offset

    return max(durations)


def _truncate_bar_overlaps(bar: ParsedBar) -> ParsedBar:
    events = sorted(bar.events, key=_event_sort_key)
    next_offsets = _next_offsets(events)
    measure_duration = Fraction(bar.time_numerator, bar.time_denominator)
    truncated_events: list[ParsedEvent] = []

    for event in events:
        next_offset = next_offsets.get(event.beat_offset)
        event_end = event.beat_offset + event.duration
        maximum_end = min(value for value in (next_offset, measure_duration) if value is not None)
        if event_end <= maximum_end:
            truncated_events.append(event)
            continue

        duration = maximum_end - event.beat_offset
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
