from __future__ import annotations

import importlib


def test_n_gram_analysis_imports() -> None:
    module = importlib.import_module("notebooks.n_gram_analysis")

    assert module.app is not None
