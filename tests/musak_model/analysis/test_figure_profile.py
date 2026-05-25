from collections import Counter
from fractions import Fraction
from pathlib import Path

from musak_model.analysis.n_grams.config import NGramAnalysisConfig
from musak_model.analysis.n_grams.figure.schema import FigureNGram
from musak_model.analysis.n_grams.profile.artifacts import FigureArtifactPaths, figure_artifact_paths
from musak_model.analysis.n_grams.profile.builder import build_figure_profile, build_figure_sample_counts
from musak_model.analysis.n_grams.profile.extraction import extract_figure_artifacts
from musak_model.analysis.n_grams.profile.io import (
    read_figure_counts_csv,
    read_figure_profile,
    read_figure_sample_counts_jsonl,
    write_figure_counts_csv,
    write_figure_profile,
    write_figure_sample_counts_jsonl,
)
from musak_model.analysis.n_grams.profile.loading import (
    figure_profile_encoded_directory,
    load_figure_profile_artifacts,
    load_processed_figure_profile_artifacts,
)
from musak_model.analysis.n_grams.profile.schema import FigureProfileMetadata, FigureSampleCounts
from musak_model.analysis.n_grams.profile.streaming import (
    FigureBatchTask,
    FigureWorkStore,
    figure_state_key,
    figure_work_store_path,
    process_figure_batch_task,
)
from musak_model.data.schema import SegmentMetadata
from musak_model.processing.io import append_jsonl, write_json_model
from musak_model.processing.paths import ProcessedDatasetPaths
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


def test_figure_counts_csv_round_trips(tmp_path: Path) -> None:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    counts = {
        ScaleType.MAJOR: {
            Hand.RIGHT: {
                1: Counter({figure: 2}),
            }
        }
    }
    path = tmp_path / "counts.csv"

    write_figure_counts_csv(counts, path)

    assert read_figure_counts_csv(path) == counts


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
    encoded_directory = tmp_path / "processed" / "PDMX" / "encoded" / "abc"
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
    write_json_model(snapshot, encoded_directory / "tokenizer.json", overwrite=True)
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
        encoded_directory / "data-00000.jsonl",
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
        encoded_directory / "data-00000.jsonl",
    )

    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
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


def test_extract_figure_artifacts_resumes_partial_work_store(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    encoded_directory, analysis_config_path = _write_encoded_figure_inputs(
        tmp_path,
        tokenization_config=tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    config = NGramAnalysisConfig.load(analysis_config_path)
    paths = figure_artifact_paths(encoded_directory)
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    first_line = (encoded_directory / "data-00000.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with FigureWorkStore(
        figure_work_store_path(paths),
        state_key=figure_state_key(config=config, snapshot=snapshot),
        resume=False,
    ) as store:
        store.commit_batch(
            process_figure_batch_task(
                FigureBatchTask(
                    batch_index=0,
                    sample_start_index=0,
                    encoded_lines=(first_line,),
                    tokenization_config=tokenization_config,
                    min_n=config.min_n,
                    max_n=config.max_n,
                )
            )
        )

    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
        analysis_config_path=analysis_config_path,
        output_path=None,
        show_progress=False,
    )

    assert result.sample_profile_count == 2
    assert not figure_work_store_path(paths).exists()
    assert len(read_figure_sample_counts_jsonl(paths.by_sample_path)) == 2


def test_extract_figure_artifacts_rejects_stale_partial_work(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    encoded_directory, analysis_config_path = _write_encoded_figure_inputs(
        tmp_path,
        tokenization_config=tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    config = NGramAnalysisConfig.load(analysis_config_path)
    paths = figure_artifact_paths(encoded_directory)
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    with FigureWorkStore(
        figure_work_store_path(paths),
        state_key=figure_state_key(config=config, snapshot=snapshot),
        resume=False,
    ):
        pass

    try:
        extract_figure_artifacts(
            encoded_directory=encoded_directory,
            analysis_config_path=_stale_analysis_config_path(tmp_path),
            output_path=None,
            show_progress=False,
        )
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected stale partial figure work to raise")

    assert "does not match" in message


def test_extract_figure_artifacts_parallel_writes_expected_profiles(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    encoded_directory, analysis_config_path = _write_encoded_figure_inputs(
        tmp_path,
        tokenization_config=tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        workers=2,
    )

    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
        analysis_config_path=analysis_config_path,
        output_path=None,
        show_progress=False,
    )

    assert result.encoded_sample_count == 2
    assert result.sample_profile_count == 2
    assert read_figure_profile(result.artifact_paths.profile_path).metadata.sample_count == 2


def test_figure_artifact_paths_resolve_under_encoded_run() -> None:
    paths = figure_artifact_paths(Path("processed/PDMX/encoded/abc"))

    assert paths.root_directory == Path("processed/PDMX/encoded/abc/figure")
    assert paths.config_path == Path("processed/PDMX/encoded/abc/figure/config.yml")
    assert paths.profile_path == Path("processed/PDMX/encoded/abc/figure/all/profile.json")
    assert paths.counts_path == Path("processed/PDMX/encoded/abc/figure/all/counts.csv")
    assert paths.by_sample_path == Path("processed/PDMX/encoded/abc/figure/by_sample.jsonl")


def test_load_figure_profile_artifacts_returns_none_when_optional_artifacts_are_missing(tmp_path: Path) -> None:
    assert load_figure_profile_artifacts(tmp_path / "encoded") is None


def test_load_figure_profile_artifacts_requires_complete_canonical_artifacts(tmp_path: Path) -> None:
    encoded_directory = tmp_path / "encoded"
    paths = figure_artifact_paths(encoded_directory)
    paths.config_path.parent.mkdir(parents=True)
    paths.config_path.write_text("min_n: 1\n", encoding="utf-8")

    try:
        load_figure_profile_artifacts(encoded_directory)
    except FileNotFoundError as error:
        message = str(error)
    else:
        raise AssertionError("expected incomplete artifacts to raise FileNotFoundError")

    assert "profile.json" in message
    assert "counts.csv" in message
    assert "by_sample.jsonl" in message


def test_load_figure_profile_artifacts_required_missing_artifacts_raise(tmp_path: Path) -> None:
    try:
        load_figure_profile_artifacts(tmp_path / "encoded", required=True)
    except FileNotFoundError as error:
        message = str(error)
    else:
        raise AssertionError("expected required artifacts to raise FileNotFoundError")

    assert "Figure profile artifacts are incomplete" in message


def test_load_figure_profile_artifacts_loads_and_validates_complete_artifacts(tmp_path: Path) -> None:
    encoded_directory = tmp_path / "encoded"
    profile = build_figure_profile(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 2}),
                }
            }
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=1),
    )
    sample_counts = build_figure_sample_counts(
        sample_index=0,
        scale_type=ScaleType.MAJOR,
        counts_by_hand={
            Hand.RIGHT: {
                1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 2}),
            }
        },
    )
    paths = figure_artifact_paths(encoded_directory)
    _write_required_artifact_placeholders(paths)
    write_figure_profile(profile, paths.profile_path)
    write_figure_sample_counts_jsonl([sample_counts], paths.by_sample_path)

    artifacts = load_figure_profile_artifacts(encoded_directory)

    assert artifacts is not None
    assert artifacts.paths == paths
    assert artifacts.profile == profile
    assert artifacts.sample_counts == (sample_counts,)


def test_load_figure_profile_artifacts_validates_sample_count(tmp_path: Path) -> None:
    encoded_directory = tmp_path / "encoded"
    profile = build_figure_profile(
        {},
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=1),
    )
    paths = figure_artifact_paths(encoded_directory)
    _write_required_artifact_placeholders(paths)
    write_figure_profile(profile, paths.profile_path)
    write_figure_sample_counts_jsonl([], paths.by_sample_path)

    try:
        load_figure_profile_artifacts(encoded_directory)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("expected inconsistent sample count to raise ValueError")

    assert "sample_count=1" in message


def test_figure_profile_encoded_directory_resolves_current_tokenizer_hash(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    processed_root = tmp_path / "processed"
    dataset_root = tmp_path / "PDMX"
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    expected_dir = ProcessedDatasetPaths.from_dataset_root(
        processed_root=processed_root,
        dataset_root=dataset_root,
    ).encoded_directory(snapshot.tokenizer_hash)

    assert (
        figure_profile_encoded_directory(
            processed_root=processed_root,
            dataset_root=dataset_root,
            tokenization_config=tokenization_config,
        )
        == expected_dir
    )


def test_load_processed_figure_profile_artifacts_infers_encoded_directory(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
) -> None:
    processed_root = tmp_path / "processed"
    dataset_root = tmp_path / "PDMX"
    encoded_directory = figure_profile_encoded_directory(
        processed_root=processed_root,
        dataset_root=dataset_root,
        tokenization_config=tokenization_config,
    )
    profile = build_figure_profile(
        {},
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=0),
    )
    paths = figure_artifact_paths(encoded_directory)
    _write_required_artifact_placeholders(paths)
    write_figure_profile(profile, paths.profile_path)
    write_figure_sample_counts_jsonl([], paths.by_sample_path)

    artifacts = load_processed_figure_profile_artifacts(
        processed_root=processed_root,
        dataset_root=dataset_root,
        tokenization_config=tokenization_config,
    )

    assert artifacts is not None
    assert artifacts.paths == paths
    assert artifacts.profile == profile


def _write_required_artifact_placeholders(paths: FigureArtifactPaths) -> None:
    paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text("min_n: 1\nmax_n: 1\n", encoding="utf-8")
    paths.counts_path.parent.mkdir(parents=True, exist_ok=True)
    paths.counts_path.write_text("scale_type,hand,n,count,figure\n", encoding="utf-8")


def _write_encoded_figure_inputs(
    tmp_path: Path,
    *,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    workers: int = 1,
) -> tuple[Path, Path]:
    encoded_directory = tmp_path / "processed" / "PDMX" / "encoded" / "abc"
    analysis_config_path = tmp_path / "n_grams.yml"
    analysis_config_path.write_text(
        "\n".join(
            [
                "min_n: 2",
                "max_n: 2",
                "limit_per_group: null",
                f"workers: {workers}",
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
    write_json_model(snapshot, encoded_directory / "tokenizer.json", overwrite=True)
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    for hand, degree, scale_type in (
        (Hand.RIGHT, 2, ScaleType.MAJOR),
        (Hand.LEFT, 3, ScaleType.HARMONIC_MINOR),
    ):
        append_jsonl(
            _sample(
                token_vocabulary.encode(
                    [
                        HandToken(hand=hand),
                        _note(1, duration_id=quarter_id),
                        _note(degree, duration_id=quarter_id),
                    ]
                ),
                scale_type=scale_type,
            ),
            encoded_directory / "data-00000.jsonl",
        )

    return encoded_directory, analysis_config_path


def _stale_analysis_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "stale_n_grams.yml"
    path.write_text(
        "\n".join(
            [
                "min_n: 2",
                "max_n: 3",
                "limit_per_group: null",
                "workers: 1",
                "batch_size: 1",
            ]
        ),
        encoding="utf-8",
    )
    return path


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
            difficulty_level=None,
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
