from pydantic import BaseModel, ConfigDict


class IntervalDefaultSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    sequential: bool = False
    lowest_note: int = 40
    highest_note: int = 90
    tempo: int = 120


class IntervalsConfig(BaseModel):
    intervals_definitions: dict[str, int]
    default_settings: IntervalDefaultSettings


class InversionDefaultSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    sequential: bool = False
    lowest_note: int = 40
    highest_note: int = 90
    tempo: int = 120


class InversionsConfig(BaseModel):
    chords_definitions: dict[str, list[int]]
    default_settings: InversionDefaultSettings


class RhythmDefaultSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    groups: int = 1
    measures: int = 2
    tempo: int = 120
    time_signature_numerator: int = 4
    time_signature_denominator: int = 4


class RhythmConfig(BaseModel):
    default_settings: RhythmDefaultSettings
