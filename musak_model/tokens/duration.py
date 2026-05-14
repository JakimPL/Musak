from fractions import Fraction

from musak_model.tokens.config import TokenizationConfig


class DurationVocabulary:
    def __init__(self, config: TokenizationConfig) -> None:
        self._config = config
        self._fractions: list[Fraction] = []
        self._fraction_to_id: dict[Fraction, int] = {}
        self._generate_durations()

    def _generate_durations(self) -> None:
        durations_set = self._generate_base_durations()
        durations_set.update(self._generate_tuplet_variants(durations_set))
        durations_set.update(self._generate_dotted_variants(durations_set))
        self._build_sorted_mapping(durations_set)

    def _build_sorted_mapping(self, durations_set: set[Fraction]) -> None:
        self._fractions = sorted(durations_set, reverse=True)
        self._fraction_to_id = {frac: idx for idx, frac in enumerate(self._fractions)}

    def _generate_base_durations(self) -> set[Fraction]:
        durations_set: set[Fraction] = set()
        shortest = self._config.shortest_duration_fraction
        current = Fraction(1, 1)
        while current >= shortest:
            durations_set.add(current)
            current /= 2

        return durations_set

    def _generate_tuplet_variants(self, durations_set: set[Fraction]) -> set[Fraction]:
        tuplet_variants: set[Fraction] = set()
        for base_duration in durations_set:
            for tuplet_divisor in self._config.max_tuplets:
                tuplet_duration = base_duration / tuplet_divisor
                tuplet_variants.add(tuplet_duration)

        return tuplet_variants

    def _generate_dotted_variants(self, durations_set: set[Fraction]) -> set[Fraction]:
        dotted_variants: set[Fraction] = set()
        for base_duration in durations_set:
            for dots in range(1, self._config.max_dots + 1):
                dotted_duration = base_duration * Fraction(2 ** (dots + 1) - 1, 2**dots)
                if dotted_duration <= Fraction(1, 1):
                    dotted_variants.add(dotted_duration)

        return dotted_variants

    def fraction_to_id(self, duration: Fraction) -> int:
        return self._fraction_to_id[duration]

    def id_to_fraction(self, duration_id: int) -> Fraction:
        return self._fractions[duration_id]

    def find_closest(self, target: Fraction) -> tuple[int, Fraction]:
        if not self._fractions:
            raise ValueError("Empty vocabulary")

        closest_frac = min(self._fractions, key=lambda f: abs(f - target))
        closest_id = self._fraction_to_id[closest_frac]
        return closest_id, closest_frac

    def all_fractions(self) -> tuple[Fraction, ...]:
        return tuple(self._fractions)

    def vocab_size(self) -> int:
        return len(self._fractions)

    def __repr__(self) -> str:
        config = self._config
        return (
            f"DurationVocabulary(shortest=1/{config.shortest_duration}, "
            f"tuplets={config.max_tuplets}, dots={config.max_dots}, "
            f"size={self.vocab_size()})"
        )
