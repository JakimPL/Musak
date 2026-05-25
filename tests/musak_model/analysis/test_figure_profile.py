from collections import Counter
from fractions import Fraction
from pathlib import Path

from musak_model.analysis.n_grams.figure.schema import FigureNGram
from musak_model.analysis.n_grams.profile.artifacts import figure_artifact_paths
from musak_model.analysis.n_grams.profile.builder import build_figure_profile, build_figure_sample_counts
from musak_model.analysis.n_grams.profile.extraction import extract_figure_artifacts
from musak_model.analysis.n_grams.profile.io import (
    read_figure_profile,
    read_figure_sample_counts_jsonl,
    write_figure_profile,
    write_figure_sample_counts_jsonl,
)
from musak_model.analysis.n_grams.profile.schema import FigureProfileMetadata, FigureSampleCounts
from musak_model.data.schema import SegmentMetadata
from musak_model.processing.io import append_jsonl, write_json_model
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def test_build_figure_profile_totals_are_deterministic() -> None:
    first = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    second = FigureNGram(onsets=((((0, 0), (2, 0)), Fraction(1)),))

    profile = build_figure_profile(
        {
            ScaleType.HARMONIC_MINOR: {Hand.RIGHT: {1: Counter({first: 2})}},
            ScaleType.MAJOR: {Hand.LEFT: {1: Counter({second: 3})}},
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=5),
    )

    assert [(group.scale_type, group.hand, group.n, group.total) for group in profile.groups] == [
        (ScaleType.HARMONIC_MINOR, Hand.RIGHT, 1, 2),
        (ScaleType.MAJOR, Hand.LEFT, 1, 3),
    ]


def test_build_figure_profile_property_totals_are_count_weighted() -> None:
    monophonic = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    chord = FigureNGram(onsets=((((0, 0), (2, 0)), Fraction(1)),))
    chromatic = FigureNGram(onsets=((((0, 1),), Fraction(1)),))

    profile = build_figure_profile(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter(
                        {
                            monophonic: 2,
                            chord: 3,
                            chromatic: 5,
                        }
                    )
                }
            }
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=1),
    )

    assert len(profile.groups) == 1
    group = profile.groups[0]
    assert group.total == 10
    assert group.monophonic == 7
    assert group.chords_only == 3
    assert group.in_scale == 5


def test_figure_profile_json_round_trips(tmp_path: Path) -> None:
    profile = build_figure_profile(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 2}),
                }
            }
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=3),
    )
    path = tmp_path / "profile.json"

    write_figure_profile(profile, path)

    assert read_figure_profile(path) == profile


def test_figure_sample_counts_jsonl_round_trips(tmp_path: Path) -> None:
    sample_counts = build_figure_sample_counts(
        sample_index=3,
        scale_type=ScaleType.MAJOR,
        counts_by_hand={
            Hand.RIGHT: {
                1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 2}),
            }
        },
    )
    path = tmp_path / "by_sample.jsonl"

    write_figure_sample_counts_jsonl([sample_counts], path)

    assert read_figure_sample_counts_jsonl(path) == [sample_counts]


def test_extract_figure_artifacts_writes_by_sample_jsonl(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    encoded_dir = tmp_path / "processed" / "PDMX" / "encoded" / "abc"
    analysis_config_path = tmp_path / "n_grams.yml"
    analysis_config_path.write_text(
        "\n".join(
            [
                "min_n: 2",
                "max_n: 2",
                "limit_per_group: null",
                "workers: 1",
                "batch_size: 1",
            ]
        ),
        encoding="utf-8",
    )
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    write_json_model(snapshot, encoded_dir / "tokenizer.json", overwrite=True)
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    append_jsonl(
        _sample(
            token_vocabulary.encode(
                [
                    HandToken(hand=Hand.RIGHT),
                    _note(1, duration_id=quarter_id),
                    _note(2, duration_id=quarter_id),
                ]
            ),
            scale_type=ScaleType.MAJOR,
        ),
        encoded_dir / "data-00000.jsonl",
    )
    append_jsonl(
        _sample(
            token_vocabulary.encode(
                [
                    HandToken(hand=Hand.LEFT),
                    _note(1, duration_id=quarter_id),
                    _note(3, duration_id=quarter_id),
                ]
            ),
            scale_type=ScaleType.HARMONIC_MINOR,
        ),
        encoded_dir / "data-00000.jsonl",
    )

    result = extract_figure_artifacts(
        encoded_dir=encoded_dir,
        analysis_config_path=analysis_config_path,
        output_path=None,
        show_progress=False,
    )
    sample_counts = read_figure_sample_counts_jsonl(result.artifact_paths.by_sample_path)

    assert result.sample_profile_count == 2
    assert result.artifact_paths.by_sample_path.is_file()
    assert [sample_count.sample_index for sample_count in sample_counts] == [0, 1]
    assert sample_counts[0].scale_type == ScaleType.MAJOR
    assert sample_counts[1].scale_type == ScaleType.HARMONIC_MINOR
    assert _group_total(sample_counts[0], hand=Hand.RIGHT) == 1
    assert _group_total(sample_counts[1], hand=Hand.LEFT) == 1


def test_figure_artifact_paths_resolve_under_encoded_run() -> None:
    paths = figure_artifact_paths(Path("processed/PDMX/encoded/abc"))

    assert paths.root_dir == Path("processed/PDMX/encoded/abc/figure")
    assert paths.config_path == Path("processed/PDMX/encoded/abc/figure/config.yml")
    assert paths.profile_path == Path("processed/PDMX/encoded/abc/figure/all/profile.json")
    assert paths.counts_path == Path("processed/PDMX/encoded/abc/figure/all/counts.csv")
    assert paths.by_sample_path == Path("processed/PDMX/encoded/abc/figure/by_sample.jsonl")


def _sample(token_ids: list[int], *, scale_type: ScaleType) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=[0 for _ in token_ids],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=scale_type,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("sample.mxl"),
        ),
    )


def _note(degree: int, *, duration_id: int) -> NoteToken:
    return NoteToken(
        degree=degree,
        accidental=0,
        octave_offset=0,
        duration_id=duration_id,
    )


def _group_total(sample_counts: FigureSampleCounts, *, hand: Hand) -> int:
    return next(group.total for group in sample_counts.groups if group.hand == hand)
