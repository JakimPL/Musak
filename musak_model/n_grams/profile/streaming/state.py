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
        "min_n": config.min_n,
        "max_n": config.max_n,
        "rhythm_min_n": config.rhythm_min_n,
        "rhythm_max_n": config.rhythm_max_n,
        "grid_alignment_denominators": config.grid_alignment_denominators,
        "strong_beat_offsets": [str(offset) for offset in config.strong_beat_offsets],
        "batch_size": config.batch_size,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
