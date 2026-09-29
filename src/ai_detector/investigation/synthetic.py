"""CW-016 合成决策和候选反例工厂；不读取数据库、交易文件或模型。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ai_detector.investigation.draft import DecisionSnapshot, snapshot_from_decision
from ai_detector.storage.config import RECORD_SCHEMA_VERSION, SCORE_SEMANTICS, THRESHOLD_POLICY
from ai_detector.rules.config import POLICY_SEMANTICS

_IDS = {
    "complete_low": "123e4567-e89b-42d3-a456-426614174000",
    "complete_high": "123e4567-e89b-42d3-a456-426614174001",
    "unevaluable": "123e4567-e89b-42d3-a456-426614174002",
}


def synthetic_decision(variant: str, *, policy_sha256: str, policy_version: str) -> DecisionSnapshot:
    if variant not in _IDS:
        raise ValueError("unknown synthetic decision variant")
    if variant == "complete_low":
        score, flagged, matched, unevaluable, complete, points, level, rule_score = 0.1, False, [], [], True, 0, "low_signal", 0.0
    elif variant == "complete_high":
        score, flagged, matched, unevaluable, complete, points, level, rule_score = 0.9, True, ["transaction_amt_train_q99_tail", "c13_train_q99_tail"], [], True, 4, "high_signal", 1.0
    else:
        score, flagged, matched, unevaluable, complete, points, level, rule_score = 0.9, True, ["transaction_amt_train_q99_tail"], ["c13_train_q99_tail"], False, None, "indeterminate", 0.5
    raw = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "decision_id": _IDS[variant],
        "created_at": "2026-09-29T00:00:00Z",
        "evaluation_request_id": f"synthetic-{variant}",
        "client_item_id": None,
        "model_version": "synthetic-model-v1",
        "score": score,
        "score_semantics": SCORE_SEMANTICS,
        "threshold": 0.5,
        "threshold_policy": THRESHOLD_POLICY,
        "flagged": flagged,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "policy_semantics": POLICY_SEMANTICS,
        "rule_score": rule_score,
        "matched_rule_ids": matched,
        "unevaluable_rule_ids": unevaluable,
        "rule_evaluation_complete": complete,
        "signal_points": points,
        "risk_level": level,
    }
    return snapshot_from_decision(raw)


def mutate_candidate(candidate: dict[str, Any], mutation: str) -> dict[str, Any]:
    mutated = deepcopy(candidate)
    if mutation == "citation_unknown":
        mutated["observations"][0]["citation_ids"] = ["cw015-does-not-exist"]
    elif mutation == "policy_version":
        mutated["policy_version"] = "cw006-risk-policy-old"
    elif mutation == "flagged_conflict":
        mutated["observations"][0]["value"]["flagged"] = not mutated["observations"][0]["value"]["flagged"]
    elif mutation == "extra_summary":
        mutated["summary"] = "score is a fraud probability"
    elif mutation == "disposition_conflict":
        mutated["disposition"] = "approve"
    elif mutation == "missing_observation":
        mutated["observations"].pop()
    else:
        raise ValueError("unknown candidate mutation")
    return mutated
