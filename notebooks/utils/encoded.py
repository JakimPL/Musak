from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from musak_model.data.schema import Segment
from musak_model.decoder import encoded_exercise_to_segment
from musak_model.processing.io import load_encoded_jsonl, load_tokenizer_snapshot_json
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.snapshot import TokenizerSnapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


@dataclass(frozen=True)
class EncodedShard:
    path: Path
    samples: list[EncodedExercise]
    snapshot: TokenizerSnapshot
    duration_vocabulary: DurationVocabulary
    token_vocabulary: TokenVocabulary


def load_encoded_shard(path: Path) -> EncodedShard:
    snapshot_path = path.parent / "tokenizer.json"
    snapshot = load_tokenizer_snapshot_json(snapshot_path)
    tokenization_config = TokenizationConfig.model_validate(snapshot.tokenization_config)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    return EncodedShard(
        path=path,
        samples=load_encoded_jsonl(path),
        snapshot=snapshot,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )


def encoded_sample_to_segment(sample: EncodedExercise, *, shard: EncodedShard) -> Segment:
    return encoded_exercise_to_segment(sample, token_vocabulary=shard.token_vocabulary)


def default_encoded_browser_root(processed_root: Path, dataset_root: Path) -> Path:
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    return paths.root / "encoded"
