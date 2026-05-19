from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_shared.time_signature import validate_time_denominator


class TimeSignatureVocabularyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_denominator: int = Field(gt=0)
    relative_numerator_range: float = Field(gt=1)

    @field_validator("max_denominator")
    @classmethod
    def _validate_max_denominator_power_of_two(cls, value: int) -> int:
        validate_time_denominator(value)
        return value


class TimeSignatureVocabulary:
    def __init__(self, config: TimeSignatureVocabularyConfig) -> None:
        self._config = config
        self._time_signatures = self._generate_time_signatures(config)
        self._time_signature_to_id = {
            time_signature: index for index, time_signature in enumerate(self._time_signatures)
        }

    @property
    def vocabulary_size(self) -> int:
        return len(self._time_signatures)

    def time_signature_to_id(self, time_signature: tuple[int, int]) -> int:
        try:
            return self._time_signature_to_id[time_signature]
        except KeyError as exception:
            raise ValueError(f"unsupported time signature: {time_signature}") from exception

    def contains(self, time_signature: tuple[int, int]) -> bool:
        return time_signature in self._time_signature_to_id

    def id_to_time_signature(self, time_signature_id: int) -> tuple[int, int]:
        if not 0 <= time_signature_id < self.vocabulary_size:
            raise KeyError(f"time_signature_id must be in [0, {self.vocabulary_size - 1}]")

        return self._time_signatures[time_signature_id]

    def all_time_signatures(self) -> tuple[tuple[int, int], ...]:
        return self._time_signatures

    @staticmethod
    def _generate_time_signatures(config: TimeSignatureVocabularyConfig) -> tuple[tuple[int, int], ...]:
        denominators: list[int] = []
        denominator = 1
        while denominator <= config.max_denominator:
            denominators.append(denominator)
            denominator *= 2

        return tuple(
            (numerator, denominator)
            for denominator in denominators
            for numerator in range(1, int(denominator * config.relative_numerator_range))
        )

    def __repr__(self) -> str:
        return (
            "TimeSignatureVocabulary("
            f"max_denominator={self._config.max_denominator}, "
            f"relative_numerator_range={self._config.relative_numerator_range}, "
            f"size={self.vocabulary_size})"
        )
