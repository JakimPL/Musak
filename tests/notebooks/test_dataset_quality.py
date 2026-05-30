from __future__ import annotations

import importlib


def test_dataset_quality_imports() -> None:
    module = importlib.import_module("notebooks.dataset_quality")

    assert module.app is not None
