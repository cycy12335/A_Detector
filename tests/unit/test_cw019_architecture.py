"""CW-019 architecture documentation contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _validator_module():
    path = ROOT / "scripts" / "validate_architecture_docs.py"
    spec = importlib.util.spec_from_file_location("validate_architecture_docs", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cw019_architecture_document_is_auditable() -> None:
    result = _validator_module().validate()
    assert result["ok"], result["errors"]
    assert result["diagram_count"] >= 3
