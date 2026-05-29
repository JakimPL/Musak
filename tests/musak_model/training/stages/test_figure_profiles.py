from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import SegmentMetadata
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit
from musak_model.training.stages.figure_profiles import split_figure_profile_metrics


def test_split_figure_profile_metrics_compare_matching_train_and_validation(tmp_path: Path) -> None:
    tokenization_config = _tokenization_config()
    token_vocabulary = _token_vocabulary(tokenization_config)
    sample = _sample(token_vocabulary, accidental=0)

    metrics = split_figure_profile_metrics(
        IngestionSplit(train=[sample], validation=[sample], invalid_files=[]),
        token_vocabulary=token_vocabulary,
        tokenization_config=tokenization_config,
        analysis_config_path=_analysis_config_path(tmp_path),
        artifact_root=tmp_path / "figure-splits",
        workers=1,
    )

    assert metrics["model/split/figure/count/train_samples"] == 1.0
    assert metrics["model/split/figure/count/validation_samples"] == 1.0
    assert metrics["model/split/figure/count/comparable_groups"] == 1.0
    assert metrics["model/split/figure/mean/total_relative_abs_error"] == 0.0
    assert metrics["model/split/figure/mean/identity_total_variation_distance"] == 0.0
    assert len(list((tmp_path / "figure-splits").glob("*/train/all/counts.parquet"))) == 1
    assert len(list((tmp_path / "figure-splits").glob("*/validation/all/counts.parquet"))) == 1

    reused_metrics = split_figure_profile_metrics(
        IngestionSplit(train=[sample], validation=[sample], invalid_files=[]),
        token_vocabulary=token_vocabulary,
        tokenization_config=tokenization_config,
        analysis_config_path=_analysis_config_path(tmp_path),
        artifact_root=tmp_path / "figure-splits",
        workers=1,
    )

    assert reused_metrics == metrics


def test_split_figure_profile_metrics_detect_different_validation_figures(tmp_path: Path) -> None:
    tokenization_config = _tokenization_config()
    token_vocabulary = _token_vocabulary(tokenization_config)

    metrics = split_figure_profile_metrics(
        IngestionSplit(
            train=[_sample(token_vocabulary, accidental=0)],
            validation=[_sample(token_vocabulary, accidental=1)],
            invalid_files=[],
        ),
        token_vocabulary=token_vocabulary,
        tokenization_config=tokenization_config,
        analysis_config_path=_analysis_config_path(tmp_path),
        artifact_root=tmp_path / "figure-splits",
        workers=1,
    )

    assert metrics["model/split/figure/count/comparable_groups"] == 1.0
    assert metrics["model/split/figure/mean/identity_total_variation_distance"] == 1.0


def test_split_figure_profile_metrics_handles_empty_validation_split(tmp_path: Path) -> None:
    tokenization_config = _tokenization_config()
    token_vocabulary = _token_vocabulary(tokenization_config)

    metrics = split_figure_profile_metrics(
        IngestionSplit(train=[_sample(token_vocabulary, accidental=0)], validation=[], invalid_files=[]),
        token_vocabulary=token_vocabulary,
        tokenization_config=tokenization_config,
        analysis_config_path=_analysis_config_path(tmp_path),
        artifact_root=tmp_path / "figure-splits",
        workers=0,
    )

    assert metrics["model/split/figure/count/train_samples"] == 1.0
    assert metrics["model/split/figure/count/validation_samples"] == 0.0
    assert metrics["model/split/figure/count/comparable_groups"] == 0.0
    assert metrics["model/split/figure/count/distribution_groups"] == 1.0


def _analysis_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "n_grams.yml"
    path.write_text(
        "\n".join(
            [
                "min_n: 1",
                "max_n: 1",
                "limit_per_group: null",
                "workers: 1",
                "batch_size: 8",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1)


def _token_vocabulary(tokenization_config: TokenizationConfig) -> TokenVocabulary:
    return TokenVocabulary(DurationVocabulary(tokenization_config))


def _sample(token_vocabulary: TokenVocabulary, *, accidental: int) -> EncodedExercise:
    quarter_id = token_vocabulary.duration_vocabulary.fraction_to_id(Fraction(1, 4))
    return EncodedExercise(
        token_ids=token_vocabulary.encode(
            [
                HandToken(hand=Hand.RIGHT),
                NoteToken(degree=1, accidental=accidental, octave_offset=0, duration_id=quarter_id),
            ]
        ),
        bar_positions=[0, 0],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
            difficulty_level=None,
        ),
    )
