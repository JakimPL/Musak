from __future__ import annotations

import importlib


def test_model_output_explorer_imports() -> None:
    module = importlib.import_module("notebooks.model_output_explorer")

    assert module.app is not None
