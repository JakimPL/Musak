from __future__ import annotations

from fractions import Fraction

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedEvent, ParsedNote, ParsedScore


def clean_parsed_score(score: ParsedScore) -> ParsedScore:
    return trim_silent_edge_bars(deduplicate_simultaneous_pitches(score))


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
