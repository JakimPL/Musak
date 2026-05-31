from collections import Counter
from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.harmony.decoding.config import ChordDecoderConfig
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.artifacts import figure_artifact_paths
from musak_model.n_grams.profile.chord.io import read_chord_metadata, read_chord_transitions
from musak_model.n_grams.profile.chord.schema import (
    INITIAL_CHORD_SOURCE,
    ChordDecodeSpec,
    chord_artifact_paths_for_figure_root,
)
from musak_model.n_grams.profile.extraction import extract_figure_artifacts
from musak_model.n_grams.profile.io import (
    read_base_duration_counts,
    read_figure_counts,
    read_figure_profile,
    read_figure_sample_counts_jsonl,
)
from musak_model.n_grams.profile.reference import FigureReferenceStore
from musak_model.n_grams.profile.rhythm.io import read_rhythm_counts, read_rhythm_profile
from musak_model.n_grams.profile.rhythm.schema import rhythm_artifact_paths_for_figure_root
from musak_model.n_grams.profile.schema import FigureSampleCounts
from musak_model.n_grams.profile.streaming.schema import FigureBatchTask
from musak_model.n_grams.profile.streaming.state import figure_state_key
from musak_model.n_grams.profile.streaming.store import FigureWorkStore, figure_reference_database_path
from musak_model.n_grams.profile.streaming.worker import process_figure_batch_task
from musak_model.processing.io import append_jsonl, write_json_model
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def test_extract_figure_artifacts_writes_by_sample_jsonl(
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

    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
        analysis_config_path=analysis_config_path,
        output_path=None,
        show_progress=False,
    )
    sample_counts = read_figure_sample_counts_jsonl(result.artifact_paths.by_sample_path)
    rhythm_paths = rhythm_artifact_paths_for_figure_root(result.artifact_paths.root_directory)

    assert result.sample_profile_count == 2
    assert result.artifact_paths.by_sample_path.is_file()
    assert rhythm_paths.counts_path.is_file()
    assert rhythm_paths.profile_path.is_file()
    assert read_rhythm_profile(rhythm_paths.profile_path).metadata.sample_count == 2
    assert read_rhythm_counts(rhythm_paths.counts_path)
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
        figure_reference_database_path(paths),
        state_key=figure_state_key(config=config, snapshot=snapshot, chord_decode=_chord_decode_spec()),
        resume=False,
    ) as store:
        store.commit_batch(
            process_figure_batch_task(
                FigureBatchTask(
                    batch_index=0,
                    sample_start_index=0,
                    encoded_lines=(first_line,),
                    tokenization_config=tokenization_config,
                    min_n=config.figure.min_n,
                    max_n=config.figure.max_n,
                    rhythm_min_n=config.rhythm.min_n,
                    rhythm_max_n=config.rhythm.max_n,
                    grid_alignment_denominators=config.rhythm.grid_alignment_denominators,
                    strong_beat_offsets=config.rhythm.strong_beat_offsets,
                    register_arch_basis_count=config.register.arch_basis_count,
                    chord_decode=_chord_decode_spec(),
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
    assert figure_reference_database_path(paths).is_file()
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
        figure_reference_database_path(paths),
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


def test_extract_figure_artifacts_populates_reference_database(
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

    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
        analysis_config_path=analysis_config_path,
        output_path=None,
        show_progress=False,
    )

    database_path = figure_reference_database_path(result.artifact_paths)
    with FigureReferenceStore(database_path) as reference:
        assert reference.anchor_counts(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT) == Counter({(1, 0, 0): 1})
        assert reference.base_duration_counts(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2) == Counter(
            {"1/4": 1}
        )
        matched = reference.figure_counts(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2, anchor_degree=1)
        unmatched = reference.figure_counts(
            scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, figure_length=2, anchor_degree=5
        )
        assert sum(matched.values()) == 1
        assert sum(unmatched.values()) == 0


def test_extract_figure_artifacts_writes_chord_substore(
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

    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
        analysis_config_path=analysis_config_path,
        output_path=None,
        show_progress=False,
    )

    chord_paths = chord_artifact_paths_for_figure_root(result.artifact_paths.root_directory)
    assert chord_paths.transitions_path.is_file()
    assert chord_paths.figure_path.is_file()
    transitions = read_chord_transitions(chord_paths.transitions_path)
    assert any(key.source_chord == INITIAL_CHORD_SOURCE for key in transitions)
    assert read_chord_metadata(chord_paths.metadata_path).sample_count == 2


def test_extract_figure_artifacts_writes_base_durations_csv(
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

    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
        analysis_config_path=analysis_config_path,
        output_path=None,
        show_progress=False,
    )

    assert result.artifact_paths.base_durations_path.is_file()
    counts_by_group = read_base_duration_counts(result.artifact_paths.base_durations_path)
    assert counts_by_group[(ScaleType.MAJOR, Hand.RIGHT, 2)] == Counter({Fraction(1, 4): 1})
    assert counts_by_group[(ScaleType.HARMONIC_MINOR, Hand.LEFT, 2)] == Counter({Fraction(1, 4): 1})


def test_counts_csv_aggregates_same_figure_across_anchors(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    encoded_directory = tmp_path / "processed" / "PDMX" / "encoded" / "anchors"
    analysis_config_path = tmp_path / "n_grams.yml"
    analysis_config_path.write_text(_analysis_config_yaml(min_n=2, max_n=2, workers=1), encoding="utf-8")
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
                    _note(3, duration_id=quarter_id),
                ]
            ),
            scale_type=ScaleType.MAJOR,
        ),
        encoded_directory / "data-00000.jsonl",
    )

    result = extract_figure_artifacts(
        encoded_directory=encoded_directory,
        analysis_config_path=analysis_config_path,
        output_path=None,
        show_progress=False,
    )

    counts = read_figure_counts(result.artifact_paths.counts_path)
    ascending_step_figures = counts[ScaleType.MAJOR][Hand.RIGHT][2]
    assert len(ascending_step_figures) == 1
    ((figure, count),) = ascending_step_figures.items()
    assert str(figure) == "0(1) +1(1)"
    assert count == 2


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
    analysis_config_path.write_text(_analysis_config_yaml(min_n=2, max_n=2, workers=workers), encoding="utf-8")
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


def _chord_decode_spec() -> ChordDecodeSpec:
    return ChordDecodeSpec(decoder_config=ChordDecoderConfig.load(), vocabulary=ChordVocabularyConfig.load())


def _stale_analysis_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "stale_n_grams.yml"
    path.write_text(_analysis_config_yaml(min_n=2, max_n=3, workers=1), encoding="utf-8")
    return path


def _analysis_config_yaml(*, min_n: int, max_n: int, workers: int) -> str:
    return "\n".join(
        [
            "figure:",
            f"  min_n: {min_n}",
            f"  max_n: {max_n}",
            "  limit_per_group: null",
            "  common_mass_threshold: 0.80",
            "rhythm:",
            "  min_n: 2",
            "  max_n: 2",
            "  grid_alignment_denominators:",
            "    - 1",
            "    - 2",
            "    - 4",
            "  strong_beat_offsets:",
            "    - 0",
            "register:",
            "  arch_basis_count: 3",
            "execution:",
            f"  workers: {workers}",
            "  batch_size: 1",
        ]
    )


def _sample(token_ids: list[int], *, scale_type: ScaleType) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=[0 for _ in token_ids],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=scale_type,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=scale_type),
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


def _group_total(sample_counts: FigureSampleCounts, *, hand: Hand) -> int:
    return next(group.total for group in sample_counts.groups if group.hand == hand)
