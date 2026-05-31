from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field

from musak_model.paths import METRICAL_GRAMMAR_CONFIG_PATH
from musak_shared.files import load_yaml_config
from musak_shared.misc import prime_factors


class MetricalLeafType(StrEnum):
    SOUND = "sound"
    TIE = "tie"
    REST = "rest"


@dataclass(frozen=True)
class MetricalNode:
    offset: Fraction
    duration: Fraction
    weight: float
    children: tuple[MetricalNode, ...] = ()
    leaf_type: MetricalLeafType | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")

        if not 0.0 < self.weight <= 1.0:
            raise ValueError("weight must lie in (0, 1]")

        if self.children:
            if self.leaf_type is not None:
                raise ValueError("internal nodes must not carry a leaf_type")

            spanned = sum((child.duration for child in self.children), Fraction(0))
            if spanned != self.duration:
                raise ValueError("children durations must sum to the node duration")

            expected_offset = self.offset
            for child in self.children:
                if child.offset != expected_offset:
                    raise ValueError("children must tile the node contiguously")
                expected_offset += child.duration
        elif self.leaf_type is None:
            raise ValueError("a leaf must carry a leaf_type")

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> tuple[MetricalNode, ...]:
        if self.is_leaf:
            return (self,)

        return tuple(leaf for child in self.children for leaf in child.leaves())

    def frontier(self, slot_duration: Fraction) -> tuple[MetricalNode, ...]:
        if self.is_leaf or self.duration <= slot_duration:
            return (self,)

        return tuple(node for child in self.children for node in child.frontier(slot_duration))


@dataclass(frozen=True)
class MetricalTree:
    time_numerator: int
    time_denominator: int
    bars: tuple[MetricalNode, ...]

    @property
    def bar_duration(self) -> Fraction:
        return Fraction(self.time_numerator, self.time_denominator)

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    @property
    def duration(self) -> Fraction:
        return self.bar_duration * len(self.bars)

    def leaves(self) -> tuple[MetricalNode, ...]:
        return tuple(leaf for bar in self.bars for leaf in bar.leaves())

    def harmonic_frontier(self, slot_duration: Fraction) -> tuple[MetricalNode, ...]:
        if slot_duration <= 0:
            raise ValueError("slot_duration must be positive")

        return tuple(node for bar in self.bars for node in bar.frontier(slot_duration))


class MetricalGrammarConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_leaf_duration: Fraction
    subdivision_probability: float = Field(ge=0.0, le=1.0)
    subdivision_decay: float = Field(ge=0.0, le=1.0)
    rest_probability: float = Field(ge=0.0, le=1.0)
    tie_probability: float = Field(ge=0.0, le=1.0)

    @classmethod
    def load(cls, path: Path = METRICAL_GRAMMAR_CONFIG_PATH) -> MetricalGrammarConfig:
        return cls.model_validate(load_yaml_config(path))


@dataclass
class _LeafTyper:
    config: MetricalGrammarConfig
    rng: Generator
    sounding: bool = False

    def next_type(self, weight: float) -> MetricalLeafType:
        rest_probability = self.config.rest_probability * (1.0 - weight)
        draw = float(self.rng.random())
        if draw < rest_probability:
            self.sounding = False
            return MetricalLeafType.REST

        if self.sounding and draw < rest_probability + self.config.tie_probability:
            return MetricalLeafType.TIE

        self.sounding = True
        return MetricalLeafType.SOUND


@dataclass(frozen=True)
class MetricalTreeSampler:
    config: MetricalGrammarConfig

    def sample(
        self,
        *,
        time_numerator: int,
        time_denominator: int,
        bar_count: int,
        rng: Generator,
    ) -> MetricalTree:
        if time_numerator <= 0 or time_denominator <= 0:
            raise ValueError("time signature components must be positive")

        if bar_count <= 0:
            raise ValueError("bar_count must be positive")

        bar_duration = Fraction(time_numerator, time_denominator)
        factors = metrical_factors(time_numerator, time_denominator)
        typer = _LeafTyper(config=self.config, rng=rng)
        bars: list[MetricalNode] = []
        offset = Fraction(0)
        for _ in range(bar_count):
            bars.append(
                self._sample_node(
                    offset=offset,
                    duration=bar_duration,
                    weight=1.0,
                    depth=0,
                    bar_duration=bar_duration,
                    factors=factors,
                    rng=rng,
                    typer=typer,
                )
            )
            offset += bar_duration

        return MetricalTree(time_numerator, time_denominator, tuple(bars))

    def _sample_node(
        self,
        *,
        offset: Fraction,
        duration: Fraction,
        weight: float,
        depth: int,
        bar_duration: Fraction,
        factors: tuple[int, ...],
        rng: Generator,
        typer: _LeafTyper,
    ) -> MetricalNode:
        if duration <= self.config.min_leaf_duration or not self._subdivide(depth, rng):
            return MetricalNode(offset, duration, weight, (), typer.next_type(weight))

        children: list[MetricalNode] = []
        child_offset = offset
        for index, child_duration in enumerate(_split_durations(duration, factors, depth)):
            child_weight = weight if index == 0 else float(child_duration / bar_duration)
            children.append(
                self._sample_node(
                    offset=child_offset,
                    duration=child_duration,
                    weight=child_weight,
                    depth=depth + 1,
                    bar_duration=bar_duration,
                    factors=factors,
                    rng=rng,
                    typer=typer,
                )
            )
            child_offset += child_duration

        return MetricalNode(offset, duration, weight, tuple(children), None)

    def _subdivide(self, depth: int, rng: Generator) -> bool:
        probability = self.config.subdivision_probability * (self.config.subdivision_decay**depth)
        return bool(rng.random() < probability)


_BINARY_BRANCHING = 2


def _split_durations(duration: Fraction, factors: tuple[int, ...], depth: int) -> tuple[Fraction, ...]:
    branching = factors[depth] if depth < len(factors) else _BINARY_BRANCHING
    part = duration / branching
    return tuple(part for _ in range(branching))


def metrical_factors(time_numerator: int, time_denominator: int) -> tuple[int, ...]:
    if _is_compound(time_numerator, time_denominator):
        return prime_factors(time_numerator // 3) + (3,)
    return prime_factors(time_numerator)


def _is_compound(time_numerator: int, time_denominator: int) -> bool:
    return time_numerator % 3 == 0 and time_numerator >= 6 and time_denominator >= 8
