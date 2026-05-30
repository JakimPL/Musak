import hashlib
import json

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.processing.snapshot import TokenizerSnapshot


def figure_state_key(
    *,
    config: NGramAnalysisConfig,
    snapshot: TokenizerSnapshot,
) -> str:
    payload = {
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
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
