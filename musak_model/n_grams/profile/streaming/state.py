import hashlib
import json
from typing import Final

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.processing.snapshot import TokenizerSnapshot

_STATE_VERSION: Final[int] = 1


def figure_state_key(
    *,
    config: NGramAnalysisConfig,
    snapshot: TokenizerSnapshot,
) -> str:
    payload = {
        "version": _STATE_VERSION,
        "tokenizer_hash": snapshot.tokenizer_hash,
        "min_n": config.min_n,
        "max_n": config.max_n,
        "batch_size": config.batch_size,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
