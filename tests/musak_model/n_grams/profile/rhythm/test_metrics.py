from collections import Counter
from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.n_grams.profile.rhythm.extraction import count_sample_rhythm_metrics
from musak_model.n_grams.profile.rhythm.io import (
    build_rhythm_profile,
    read_rhythm_counts,
    write_rhythm_counts,
)
from musak_model.n_grams.profile.rhythm.metrics import rhythm_reference_distribution_metrics
from musak_model.n_grams.profile.rhythm.schema import (
    RhythmCountCounter,
    RhythmCountKey,
    RhythmMetricKind,
    RhythmProfileMetadata,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def test_count_sample_rhythm_metrics_extracts_musical_reference_slices(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    eighth_id = duration_vocabulary.require_duration_id(Fraction(1, 8))
    sample = _sample(
        token_vocabulary.encode(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=quarter_id),
                _note(2, duration_id=eighth_id),
                _note(3, duration_id=eighth_id),
            ]
        )
    )

    counts = count_sample_rhythm_metrics(
        sample,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        rhythm_min_n=2,
        rhythm_max_n=2,
        grid_alignment_denominators=(4,),
        strong_beat_offsets=(Fraction(0),),
    )

    assert counts[_key(kind="duration_value", parameter="", value="1/4")] == 1
    assert counts[_key(kind="duration_value", parameter="", value="1/8")] == 2
    assert counts[_key(kind="onset_grid_alignment", parameter="4", value="aligned")] == 2
    assert counts[_key(kind="onset_grid_alignment", parameter="4", value="off_grid")] == 1
    assert counts[_key(kind="duration_grid_alignment", parameter="4", value="aligned")] == 1
    assert counts[_key(kind="duration_grid_alignment", parameter="4", value="off_grid")] == 2
    assert counts[_key(kind="strong_beat_onset", parameter="", value="strong")] == 1
    assert counts[_key(kind="strong_beat_onset", parameter="", value="weak")] == 2
    assert sum(count for key, count in counts.items() if key.kind == "rhythm_ngram") == 2


def test_rhythm_counts_csv_round_trips_and_builds_profile(tmp_path: Path) -> None:
    counts: RhythmCountCounter = Counter(
        {
            _key(kind="duration_value", parameter="", value="1/4"): 2,
            _key(kind="duration_value", parameter="", value="1/8"): 1,
        }
    )
    path = tmp_path / "counts.parquet"

    write_rhythm_counts(counts, path)
    profile = build_rhythm_profile(
        read_rhythm_counts(path),
        metadata=RhythmProfileMetadata(
            rhythm_min_n=2,
            rhythm_max_n=4,
            grid_alignment_denominators=(1, 2, 4),
            strong_beat_offsets=(Fraction(0),),
            sample_count=1,
        ),
    )

    assert read_rhythm_counts(path) == counts
    assert profile.groups[0].total == 3
    assert profile.groups[0].unique_values == 2


def test_rhythm_reference_distribution_metrics_compare_alignment_and_novelty() -> None:
    reference: RhythmCountCounter = Counter(
        {
            _key(kind="duration_value", parameter="", value="1/4"): 3,
            _key(kind="duration_value", parameter="", value="1/8"): 1,
            _key(kind="strong_beat_onset", parameter="", value="strong"): 3,
            _key(kind="strong_beat_onset", parameter="", value="weak"): 1,
        }
    )
    comparison: RhythmCountCounter = Counter(
        {
            _key(kind="duration_value", parameter="", value="1/4"): 1,
            _key(kind="duration_value", parameter="", value="1/8"): 3,
            _key(kind="strong_beat_onset", parameter="", value="strong"): 1,
            _key(kind="strong_beat_onset", parameter="", value="weak"): 3,
        }
    )

    metrics = rhythm_reference_distribution_metrics(
        reference_counts=reference,
        comparison_counts=comparison,
        metric_prefix="model/generated/rhythm",
    )

    assert metrics["model/generated/rhythm/count/duration_value_distribution_groups"] == 1.0
    assert metrics["model/generated/rhythm/mean/duration_value_total_variation_distance"] == 0.5
    assert metrics["model/generated/rhythm/mean/strong_beat_onset_fraction_absolute_error"] == 0.5


def _sample(token_ids: list[int]) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=[0 for _ in token_ids],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("sample.mxl"),
            difficulty_level=None,
        ),
    )


def _note(degree: int, *, duration_id: int) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=duration_id)


def _key(*, kind: RhythmMetricKind, parameter: str, value: str) -> RhythmCountKey:
    return RhythmCountKey(
        scale_type=ScaleType.MAJOR.value,
        time_signature="4/4",
        hand=Hand.RIGHT.value,
        kind=kind,
        parameter=parameter,
        value=value,
    )
