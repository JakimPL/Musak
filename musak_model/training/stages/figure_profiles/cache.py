import hashlib
import json
from typing import Protocol

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit


class Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...


def split_cache_key(
    split: IngestionSplit,
    *,
    config: NGramAnalysisConfig,
    token_vocabulary: TokenVocabulary,
    tokenization_config: TokenizationConfig,
) -> str:
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=token_vocabulary.duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "tokenizer_hash": snapshot.tokenizer_hash,
                "min_n": config.figure_analysis.min_n,
                "max_n": config.figure_analysis.max_n,
                "rhythm_min_n": config.rhythm_analysis.min_n,
                "rhythm_max_n": config.rhythm_analysis.max_n,
                "grid_alignment_denominators": config.rhythm_analysis.grid_alignment_denominators,
                "strong_beat_offsets": [str(offset) for offset in config.rhythm_analysis.strong_beat_offsets],
                "register_arch_basis_count": config.register_analysis.arch_basis_count,
                "batch_size": config.execution.batch_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _update_samples_hash(hasher, "train", split.train)
    _update_samples_hash(hasher, "validation", split.validation)
    return hasher.hexdigest()


def _update_samples_hash(
    hasher: Hasher,
    split_name: str,
    samples: list[EncodedExercise],
) -> None:
    hasher.update(split_name.encode("utf-8"))
    hasher.update(str(len(samples)).encode("utf-8"))
    for sample in samples:
        hasher.update(sample.model_dump_json().encode("utf-8"))
