import random
from fractions import Fraction

from musak.modules.elements import misc
from musak.modules.elements.note import Note
from musak.modules.elements.phrase import Phrase
from musak.modules.rhythm.exceptions import InvalidPhraseSetError
from musak.modules.rhythm.settings import Settings


class RhythmGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: list[Phrase] | None = None

    def __call__(self) -> list[Phrase]:
        self._cache = self._generate_score()
        return self._cache

    def _generate_measure(self, group_id: int) -> Phrase:
        group_settings = self.settings.group_settings(group_id)
        phrases = group_settings.get_all_phrases()

        remainder = Fraction(*self.settings.time_signature)
        validation_message = self._validate(phrases, remainder)
        if validation_message:
            raise InvalidPhraseSetError(f"invalid set of notes/phrases, {validation_message}")

        elements = []
        while remainder:
            possible_phrases = [phrase for phrase in phrases if phrase.length <= remainder]
            choice = random.choice(possible_phrases)
            elements.append(choice)
            length = choice.length
            remainder -= length

        random.shuffle(elements)
        measure = self._flatten(elements)
        return measure

    def _generate_group(self, group_id: int) -> Phrase:
        return sum(
            [self._generate_measure(group_id) for _ in range(self.settings.measures)],
            Phrase(),
        )

    def _generate_score(self) -> list[Phrase]:
        return [self._generate_group(group_id) for group_id in range(self.settings.groups)]

    @staticmethod
    def _validate(phrases: list[Phrase], remainder: Fraction) -> str:
        gcd = Fraction(
            misc.gcd([phrase.length.numerator for phrase in phrases] + [remainder.numerator]),
            misc.lcm([phrase.length.denominator for phrase in phrases] + [remainder.denominator]),
        )

        min_length = min(phrase.length for phrase in phrases)
        if min_length > gcd:
            return f"missing notes of length {gcd}"

        if min_length > remainder:
            return f"too long notes, required a note of length {remainder}"

        return ""

    @staticmethod
    def _flatten(phrases: list[Phrase]) -> Phrase:
        all_notes: list[Note] = []
        for phrase in phrases:
            all_notes.extend(phrase.notes)

        return Phrase(notes=all_notes)

    @property
    def cache(self) -> list[Phrase] | None:
        return self._cache
