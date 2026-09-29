"""CW-015 可追溯来源、离线检索和合成评测契约。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai_detector.retrieval import RetrievalError, SourceManifestError, build_index, load_source_manifest
from scripts.evaluate_retrieval import evaluate

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/retrieval_sources.json"


def test_manifest_builds_only_verified_reviewed_sources() -> None:
    specs = load_source_manifest()
    assert {spec.authority for spec in specs} == {"policy", "explanation"}
    index = build_index()
    assert len(index.chunks) == 9
    assert index.policy_versions == ("cw006-risk-policy-v1",)
    assert all(chunk.source_path.startswith(("configs/", "README.md", "docs/")) for chunk in index.chunks)


def test_every_citation_resolves_to_exact_source_lines() -> None:
    index = build_index()
    for chunk in index.chunks:
        source = ROOT / chunk.source_path
        lines = source.read_text(encoding="utf-8").splitlines()
        assert "\n".join(lines[chunk.line_start - 1 : chunk.line_end]).strip() == chunk.text
        assert hashlib.sha256(source.read_bytes()).hexdigest() == chunk.source_sha256
        assert index.citation(chunk.chunk_id) == chunk


def test_source_tampering_fails_closed() -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["sources"][1]["sha256"] = "0" * 64
    relative_manifest = "tests/fixtures/cw015/cw015_manifest_hash_mismatch.json"
    staged_manifest = ROOT / relative_manifest
    staged_manifest.parent.mkdir(parents=True, exist_ok=True)
    staged_manifest.write_text(json.dumps(raw), encoding="utf-8")
    try:
        with pytest.raises(SourceManifestError) as error:
            build_index(relative_manifest)
    finally:
        staged_manifest.unlink(missing_ok=True)
    assert error.value.code == "source_hash_mismatch"


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("path", ".env", "source_not_allowed"),
        ("version", "unreviewed-policy-v2", "source_not_allowed"),
        ("sha256", "0" * 64, "policy_version_mismatch"),
    ],
)
def test_unreviewed_source_or_policy_version_fails_closed(field: str, value: str, expected_code: str) -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["sources"][0][field] = value
    relative_manifest = "tests/fixtures/cw015/cw015_manifest_unreviewed.json"
    staged_manifest = ROOT / relative_manifest
    staged_manifest.write_text(json.dumps(raw), encoding="utf-8")
    try:
        with pytest.raises(SourceManifestError) as error:
            build_index(relative_manifest)
    finally:
        staged_manifest.unlink(missing_ok=True)
    assert error.value.code == expected_code


def test_manifest_rejects_path_escape_and_sensitive_source() -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["sources"][0]["path"] = "../risk_policy.yaml"
    relative_manifest = "tests/fixtures/cw015/cw015_manifest_path_escape.json"
    staged_manifest = ROOT / relative_manifest
    staged_manifest.parent.mkdir(parents=True, exist_ok=True)
    staged_manifest.write_text(json.dumps(raw), encoding="utf-8")
    try:
        with pytest.raises(SourceManifestError) as error:
            load_source_manifest(relative_manifest)
    finally:
        staged_manifest.unlink(missing_ok=True)
    assert error.value.code == "source_path_unsafe"


def test_retrieval_is_deterministic_and_tie_breaks_by_citation_id() -> None:
    index = build_index()
    first = index.retrieve("C13", top_k=10)
    second = index.retrieve("C13", top_k=10)
    assert first == second
    assert [hit.chunk.chunk_id for hit in first] == sorted(hit.chunk.chunk_id for hit in first)
    assert all(hit.score > 0 for hit in first)


def test_retrieval_returns_empty_without_fabricating_evidence() -> None:
    index = build_index()
    assert index.retrieve("客户姓名 手机号码 家庭住址") == ()


def test_retrieval_limits_top_k_and_output_size() -> None:
    index = build_index()
    with pytest.raises(RetrievalError) as top_k_error:
        index.retrieve("规则", top_k=11)
    assert top_k_error.value.code == "top_k_invalid"
    with pytest.raises(RetrievalError) as output_error:
        index.retrieve("规则", max_output_chars=255)
    assert output_error.value.code == "output_limit_invalid"
    for budget in (256, 1_000, 4_000):
        hits = index.retrieve("规则 C13", top_k=10, max_output_chars=budget)
        assert len(json.dumps([hit.as_dict() for hit in hits], ensure_ascii=False)) <= budget


def test_fake_citation_is_rejected() -> None:
    index = build_index()
    hit = index.retrieve("严格", top_k=1)[0]
    hit = replace(hit, chunk=replace(hit.chunk, text="fabricated"))
    with pytest.raises(RetrievalError) as error:
        index.validate_hit(hit)
    assert error.value.code == "citation_invalid"


def test_reviewed_evaluation_is_computed_from_queries() -> None:
    report = evaluate(top_k=5)
    assert report["dataset_version"] == "cw015-synthetic-reviewed-v1"
    assert report["n_queries"] == 9
    assert report["positive_queries"] == 8
    assert report["recall_at_1"] == 0.5625
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 0.90625
    assert report["negative_empty_rate"] == 1.0

