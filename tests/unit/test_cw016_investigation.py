"""CW-016 严格草稿 schema、事实映射和引用失败关闭测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from ai_detector.investigation import (
    build_context,
    generate_candidate,
    mutate_candidate,
    snapshot_from_decision,
    synthetic_decision,
    validate_candidate,
)
from ai_detector.retrieval import build_index
from ai_detector.storage.config import RECORD_SCHEMA_VERSION, SCORE_SEMANTICS, THRESHOLD_POLICY
from ai_detector.rules.config import POLICY_SEMANTICS
from scripts.evaluate_investigation import evaluate


@pytest.fixture(scope="module")
def index():
    return build_index()


@pytest.fixture(scope="module")
def snapshot(index):
    chunk = next(item for item in index.chunks if item.authority == "policy")
    return synthetic_decision("complete_low", policy_sha256=chunk.source_sha256, policy_version=chunk.source_version)


def test_generated_complete_candidate_is_verified(index, snapshot) -> None:
    context = build_context(snapshot, index)
    result = validate_candidate(generate_candidate(snapshot, context), snapshot, context, index)
    assert result.status == "verified"
    assert result.error_codes == ()
    assert set(result.draft) == {"schema_version", "decision_id", "policy_version", "policy_sha256", "status", "disposition", "observations"}
    assert all("features" not in observation for observation in result.draft["observations"])


def test_unevaluable_rules_are_explicitly_insufficient_evidence(index) -> None:
    chunk = next(item for item in index.chunks if item.authority == "policy")
    snapshot = synthetic_decision("unevaluable", policy_sha256=chunk.source_sha256, policy_version=chunk.source_version)
    context = build_context(snapshot, index)
    result = validate_candidate(generate_candidate(snapshot, context), snapshot, context, index)
    assert result.status == "insufficient_evidence"
    assert result.accepted is True


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("citation_unknown", "citation_set_invalid"),
        ("policy_version", "policy_version_mismatch"),
        ("flagged_conflict", "fact_mismatch"),
        ("extra_summary", "draft_schema_invalid"),
        ("disposition_conflict", "draft_status_invalid"),
        ("missing_observation", "observation_count_invalid"),
        ("observation_id_claim", "observation_id_invalid"),
        ("signal_points_bool", "observation_value_invalid"),
    ],
)
def test_malicious_candidates_are_rejected(index, snapshot, mutation, code) -> None:
    context = build_context(snapshot, index)
    candidate = mutate_candidate(generate_candidate(snapshot, context), mutation)
    result = validate_candidate(candidate, snapshot, context, index)
    assert result.status == "rejected"
    assert result.draft is None
    assert result.error_codes == (code,)


def test_decision_input_rejects_raw_features_and_cross_field_conflict(index) -> None:
    chunk = next(item for item in index.chunks if item.authority == "policy")
    raw = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "decision_id": "123e4567-e89b-42d3-a456-426614174003",
        "created_at": "2026-09-29T00:00:00Z",
        "evaluation_request_id": "synthetic-invalid",
        "client_item_id": None,
        "model_version": "synthetic-model-v1",
        "score": 0.1,
        "score_semantics": SCORE_SEMANTICS,
        "threshold": 0.5,
        "threshold_policy": THRESHOLD_POLICY,
        "flagged": True,
        "policy_version": chunk.source_version,
        "policy_sha256": chunk.source_sha256,
        "policy_semantics": POLICY_SEMANTICS,
        "rule_score": 0.0,
        "matched_rule_ids": [],
        "unevaluable_rule_ids": [],
        "rule_evaluation_complete": True,
        "signal_points": 0,
        "risk_level": "low_signal",
        "features": {"C13": 1.0},
    }
    with pytest.raises(Exception) as error:
        snapshot_from_decision(raw)
    assert "features" not in str(error.value)


def test_policy_version_or_digest_drift_fails_context(index, snapshot) -> None:
    drifted = deepcopy(snapshot)
    object.__setattr__(drifted, "policy_version", "cw006-risk-policy-old")
    with pytest.raises(Exception) as error:
        build_context(drifted, index)
    assert "policy_version_mismatch" in str(error.value)


def test_evaluation_metrics_are_computed_from_reviewed_cases() -> None:
    report = evaluate()
    assert report["n_cases"] == 11
    assert report["format_cases"] == 11
    assert report["fact_cases"] == 7
    assert report["evidence_cases"] == 6
    assert report["format_success_rate"] == 1.0
    assert report["fact_consistency_rate"] == 1.0
    assert report["evidence_completeness_rate"] == 1.0
    assert report["observed_status_counts"] == {"verified": 2, "insufficient_evidence": 1, "rejected": 8}
