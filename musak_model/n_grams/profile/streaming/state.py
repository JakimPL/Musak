from typing import Any

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.chord.schema import ChordDecodeSpec
from musak_model.processing.snapshot import TokenizerSnapshot
from musak_shared.files import get_fingerprint


def figure_state_key(
    *,
    config: NGramAnalysisConfig,
    snapshot: TokenizerSnapshot,
    chord_decode: ChordDecodeSpec | None = None,
) -> str:
    payload: dict[str, Any] = {
        "tokenizer_hash": snapshot.tokenizer_hash,
        "min_n": config.figure.min_n,
        "max_n": config.figure.max_n,
        "rhythm_min_n": config.rhythm.min_n,
        "rhythm_max_n": config.rhythm.max_n,
        "grid_alignment_denominators": config.rhythm.grid_alignment_denominators,
        "strong_beat_offsets": [str(offset) for offset in config.rhythm.strong_beat_offsets],
        "register_arch_basis_count": config.register.arch_basis_count,
        "batch_size": config.execution.batch_size,
    }
    if chord_decode is not None:
        payload["chord_decoder"] = chord_decode.decoder_config.model_dump(mode="json")
        payload["chord_vocabulary"] = chord_decode.vocabulary.model_dump(mode="json")

    return get_fingerprint(payload)
