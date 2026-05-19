from __future__ import annotations

import importlib


def test_tokenizer_explorer_imports() -> None:
    module = importlib.import_module("notebooks.tokenizer_explorer")

    assert module.app is not None
