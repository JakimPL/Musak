from fractions import Fraction

from musak_model.harmony.diatonic import natural_triad
from musak_model.synthetic.fitting.form.analysis import AnalyzedPiece, HarmonicSlot
from musak_model.synthetic.fitting.form.cadence import CadenceDetectionConfig, detect_cadences
from musak_model.tokens.schema import ScaleType
from musak_shared.elements import HarmonicFunction

_CHORD = natural_triad(ScaleType.MAJOR, 1)
_BASE_CONFIG = CadenceDetectionConfig(
    metrical_weight=1.0,
    harmonic_arrival_weight=1.5,
    rhythmic_stop_weight=0.5,
    bar_alignment_weight=0.5,
    cadence_threshold=1.5,
    minimum_cadence_separation_slots=1,
    maximum_closing_slots=3,
)


def _config(**overrides: object) -> CadenceDetectionConfig:
    return _BASE_CONFIG.model_copy(update=overrides)


def _slot(
    function: HarmonicFunction | None,
    *,
    start: Fraction,
    end: Fraction,
    bar: int,
    weight: float = 0.5,
    overlap: float = 0.0,
    dwell: float = 0.5,
) -> HarmonicSlot:
    return HarmonicSlot(
        start=start,
        end=end,
        bar_index=bar,
        chord=_CHORD,
        function=function,
        metrical_weight=weight,
        tonic_triad_overlap=overlap,
        dwell=dwell,
    )


def _piece(slots: list[HarmonicSlot], *, bar_count: int) -> AnalyzedPiece:
    return AnalyzedPiece(
        scale_type=ScaleType.MAJOR,
        bar_count=bar_count,
        bar_duration=Fraction(1),
        slots=tuple(slots),
        bar_figures=(),
    )


def test_detects_authentic_cadence_with_tonic_terminal() -> None:
    slots = [
        _slot(HarmonicFunction.TONIC, start=Fraction(0), end=Fraction(1, 2), bar=0, weight=1.0, overlap=1.0),
        _slot(HarmonicFunction.PREDOMINANT, start=Fraction(1, 2), end=Fraction(1), bar=0),
        _slot(HarmonicFunction.DOMINANT, start=Fraction(1), end=Fraction(3, 2), bar=1, weight=1.0),
        _slot(HarmonicFunction.TONIC, start=Fraction(3, 2), end=Fraction(2), bar=1, overlap=1.0),
    ]

    cadences = detect_cadences(_piece(slots, bar_count=2), config=_config())

    final = cadences[-1]
    assert final.is_final
    assert final.closing.terminal_function is HarmonicFunction.TONIC
    assert final.closing.functions == (HarmonicFunction.PREDOMINANT, HarmonicFunction.DOMINANT, HarmonicFunction.TONIC)


def test_detects_half_cadence_with_dominant_terminal() -> None:
    slots = [
        _slot(HarmonicFunction.TONIC, start=Fraction(0), end=Fraction(1), bar=0, weight=1.0, overlap=1.0),
        _slot(HarmonicFunction.PREDOMINANT, start=Fraction(1), end=Fraction(3, 2), bar=1),
        _slot(HarmonicFunction.DOMINANT, start=Fraction(3, 2), end=Fraction(2), bar=1, weight=1.0, dwell=1.0),
    ]

    cadences = detect_cadences(_piece(slots, bar_count=2), config=_config())

    assert cadences[-1].closing.terminal_function is HarmonicFunction.DOMINANT


def test_final_slot_is_always_a_cadence() -> None:
    slots = [
        _slot(HarmonicFunction.TONIC, start=Fraction(0), end=Fraction(1), bar=0, weight=0.1, dwell=0.1),
        _slot(HarmonicFunction.TONIC, start=Fraction(1), end=Fraction(2), bar=1, weight=0.1, dwell=0.1),
    ]

    cadences = detect_cadences(_piece(slots, bar_count=2), config=_config(cadence_threshold=10.0))

    assert len(cadences) == 1
    assert cadences[0].is_final
    assert cadences[0].arrival_slot_index == len(slots) - 1


def test_non_maximum_suppression_thins_adjacent_arrivals() -> None:
    slots = [
        _slot(
            HarmonicFunction.DOMINANT,
            start=Fraction(index, 2),
            end=Fraction(index + 1, 2),
            bar=index // 2,
            weight=1.0,
            dwell=1.0,
        )
        for index in range(4)
    ]

    config = _config(minimum_cadence_separation_slots=1, bar_alignment_weight=0.0)
    cadences = detect_cadences(_piece(slots, bar_count=2), config=config)

    assert {cadence.arrival_slot_index for cadence in cadences} == {0, 3}
