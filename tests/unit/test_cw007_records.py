"""CW-007 决策记录的单元测试：最小化字段、严格编码与逐字段读回校验。

记录是**持久化契约**：写进去什么、读回来什么必须逐字段一致，而任何被篡改或损坏的行
（非 0/1 布尔、NaN/Infinity、非法 JSON/UUID/时间/等级、跨字段矛盾）都必须在读取时被拒绝，
**不得**被真值转换或静默修补。
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from ai_detector.rules.config import ALL_RISK_LEVELS
from ai_detector.storage.config import RECORD_SCHEMA_VERSION
from ai_detector.storage.records import (
    StoredDecision,
    StorageContractError,
    decode_rule_ids,
    encode_rule_ids,
)
from tests.cw007_fixtures import CREATED_AT, DECISION_ID_A, build_assessment


def _record(*, flagged: bool = False, matched: int = 1, **overrides: object) -> StoredDecision:
    """按 ``(flagged, matched)`` **一致地**构造记录。

    刻意不由 ``dataclasses.replace`` 改单字段——那会破坏 ``signal_points`` 与命中数的
    跨字段不变量，构造出来的就不是"合法记录被篡改"，而是"从一开始就自相矛盾"。

    ``score`` 由 ``flagged`` 推导（``0.9 >= 0.11`` / ``0.05 < 0.11``）：``flagged`` 是模型
    阈值结果，``flagged == (score >= threshold)`` 是**写入侧也成立**的跨字段不变量，
    因此这里不给调用方留下制造矛盾的机会。
    """
    assessment = build_assessment(flagged=flagged, matched=matched)
    base = StoredDecision.from_assessment(
        assessment,
        flagged=flagged,
        decision_id=DECISION_ID_A,
        created_at=CREATED_AT,
        evaluation_request_id="req-1",
        client_item_id=None,
        model_version="cw004-lightgbm-challenger-v1",
        score=0.9 if flagged else 0.05,
        threshold=0.11,
        policy_version="cw006-risk-policy-v1",
        policy_sha256="a" * 64,
    )
    return replace(base, **overrides) if overrides else base


# --- 字段集合是冻结的 ---------------------------------------------------


def test_record_carries_exactly_the_allowed_fields() -> None:
    record = _record()
    assert set(record.to_row()) == {
        "record_schema_version",
        "decision_id",
        "created_at",
        "evaluation_request_id",
        "client_item_id",
        "model_version",
        "score",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "flagged",
        "policy_version",
        "policy_sha256",
        "policy_semantics",
        "rule_score",
        "matched_rule_ids",
        "unevaluable_rule_ids",
        "rule_evaluation_complete",
        "signal_points",
        "risk_level",
    }
    assert record.record_schema_version == RECORD_SCHEMA_VERSION
    assert record.score_semantics == "uncalibrated_model_score"
    assert record.threshold_policy == "validation_f1"
    assert record.policy_semantics == "research_signal_band"


def test_record_has_no_payload_or_feature_fields() -> None:
    """记录里**不得**出现 features、特征名、请求体、header 或路径类字段。"""
    blob = json.dumps({key: str(value) for key, value in _record().to_row().items()})
    for banned in ("features", "TransactionAmt", "C13", "body", "header", "C:\\", ".yaml"):
        assert banned not in blob, banned
    assert not any("feature" in key for key in _record().to_row())


def test_record_is_immutable() -> None:
    record = _record()
    with pytest.raises(FrozenInstanceError):
        record.score = 1.0  # type: ignore[misc]


# --- 编码：SQLite 友好且确定性 ------------------------------------------


def test_row_encodes_booleans_as_integers_and_lists_as_json() -> None:
    row = _record(flagged=True).to_row()
    assert row["flagged"] == 1
    assert isinstance(row["flagged"], int) and not isinstance(row["flagged"], bool)
    assert row["rule_evaluation_complete"] == 1
    assert isinstance(row["matched_rule_ids"], str)
    assert json.loads(row["matched_rule_ids"]) == ["transaction_amt_train_q99_tail"]


def test_bool_encoding_is_strict() -> None:
    assert _record(flagged=False).to_row()["flagged"] == 0
    assert _record(flagged=True).to_row()["flagged"] == 1


@pytest.mark.parametrize(
    "ids", [(), ("a",), ("a", "b"), ("rule with spaces", "rule_2"), ("中文规则",)]
)
def test_rule_id_json_round_trip_is_deterministic(ids: tuple[str, ...]) -> None:
    encoded = encode_rule_ids(ids)
    assert decode_rule_ids(encoded) == ids
    assert encode_rule_ids(ids) == encode_rule_ids(ids)


def test_rule_id_encoding_is_compact_and_ordered() -> None:
    assert encode_rule_ids(("b", "a")) == '["b","a"]'


# --- 读取时严格校验：损坏行不得被接受 ----------------------------------


@pytest.mark.parametrize("value", [2, -1, "1", "true", None, 1.5, True])
def test_non_boolean_flagged_row_is_rejected(value: object) -> None:
    row = _record().to_row()
    row["flagged"] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


@pytest.mark.parametrize("value", [2, -1, "0", None, 0.5, False])
def test_non_boolean_complete_row_is_rejected(value: object) -> None:
    row = _record().to_row()
    row["rule_evaluation_complete"] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


@pytest.mark.parametrize("field", ["score", "threshold", "rule_score"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "0.5", None])
def test_non_finite_numbers_are_rejected(field: str, value: object) -> None:
    row = _record().to_row()
    row[field] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


@pytest.mark.parametrize("value", [-0.5, 1.5, 2.0])
def test_rule_score_outside_unit_interval_is_rejected(value: float) -> None:
    row = _record().to_row()
    row["rule_score"] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


@pytest.mark.parametrize(
    "value",
    [
        "not-a-uuid",
        "11111111111141118111111111111111",
        "11111111-1111-4111-8111-11111111111",
        "11111111-1111-4111-8111-1111111111111",
        "11111111-1111-4111-8111-11111111111Z",
        "11111111-1111-4111-8111-11111111111 ",
        "AAAAAAAA-1111-4111-8111-111111111111",  # 必须是小写
        "11111111-1111-3111-8111-111111111111",  # 必须 version 4
        "11111111-1111-4111-1111-111111111111",  # 必须 variant 8/9/a/b
        "",
    ],
)
def test_invalid_decision_id_is_rejected(value: str) -> None:
    row = _record().to_row()
    row["decision_id"] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-25T04:05:06",
        "2026-09-25 04:05:06Z",
        "2026-09-25T04:05:06+00:00",
        "2026-09-25T04:05:06z",
        "not-a-time",
        "",
        "2026-13-25T04:05:06Z",
    ],
)
def test_invalid_created_at_is_rejected(value: str) -> None:
    row = _record().to_row()
    row["created_at"] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


@pytest.mark.parametrize("value", ["unknown_signal", "LOW_SIGNAL", "", "low", None, 5])
def test_invalid_risk_level_is_rejected(value: object) -> None:
    row = _record().to_row()
    row["risk_level"] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


@pytest.mark.parametrize("value", ["not json", '{"a": 1}', '["a", 1]', '"a"', "[[]]", "[null]"])
def test_invalid_rule_id_json_is_rejected(value: str) -> None:
    row = _record().to_row()
    row["matched_rule_ids"] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


def test_duplicate_rule_ids_are_rejected() -> None:
    row = _record().to_row()
    row["matched_rule_ids"] = '["a","a"]'
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


def test_record_schema_version_mismatch_is_rejected() -> None:
    row = _record().to_row()
    row["record_schema_version"] = "cw007_decision_record_v2"
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_required_text_is_rejected(value: object) -> None:
    row = _record().to_row()
    row["model_version"] = value
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


def test_unknown_column_is_rejected() -> None:
    row = _record().to_row()
    row["unexpected_column"] = 1
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


def test_missing_column_is_rejected() -> None:
    row = _record().to_row()
    row.pop("risk_level")
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


# --- 跨字段不变量 -------------------------------------------------------


def test_signal_points_must_match_flagged_and_match_count() -> None:
    """融合真值表是冻结的：``signal_points`` 必须等于 ``flagged`` + 命中数。"""
    row = _record(flagged=False).to_row()
    row["signal_points"] = 2  # 未标记且只命中 1 条 -> 应为 1
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


def test_risk_level_must_match_signal_points() -> None:
    row = _record(flagged=False).to_row()
    row["risk_level"] = "high_signal"  # 1 点 -> 应为 elevated_signal
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


def test_indeterminate_requires_null_points_and_unevaluable_ids() -> None:
    assessment = build_assessment(flagged=False, matched=0, complete=False)
    record = StoredDecision.from_assessment(
        assessment,
        flagged=False,
        decision_id=DECISION_ID_A,
        created_at=CREATED_AT,
        evaluation_request_id="req-1",
        client_item_id=None,
        model_version="m1",
        score=0.0,
        threshold=0.11,
        policy_version="cw006-risk-policy-v1",
        policy_sha256="a" * 64,
    )
    row = record.to_row()
    assert row["risk_level"] == "indeterminate"
    assert row["signal_points"] is None
    assert json.loads(row["unevaluable_rule_ids"])

    # 给 indeterminate 补一个点数 -> 拒绝
    broken = dict(row)
    broken["signal_points"] = 1
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(broken)


def test_complete_record_must_not_have_unevaluable_ids() -> None:
    row = _record().to_row()
    row["unevaluable_rule_ids"] = '["c13_train_q99_tail"]'
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


def test_matched_and_unevaluable_must_be_disjoint() -> None:
    row = _record().to_row()
    row["unevaluable_rule_ids"] = json.dumps(json.loads(row["matched_rule_ids"]))
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


# --- flagged 是模型阈值结果，不是从可空点数反推出来的 -------------------


def test_flagged_must_equal_the_score_threshold_comparison() -> None:
    """``flagged == (score >= threshold)``：构造时就不允许自相矛盾。

    ``flagged`` 是模型的阈值判定，即使规则不可评估也**仍然已知**；它必须由调用方显式传入，
    不能从 ``signal_points`` 这类可空字段反推。
    """
    with pytest.raises(StorageContractError):
        _record(flagged=True, score=0.05)  # 标记为真但分数低于阈值
    with pytest.raises(StorageContractError):
        _record(flagged=False, score=0.90)  # 未标记但分数高于阈值


def test_exactly_on_the_threshold_counts_as_flagged() -> None:
    """边界是 ``>=``：分数恰好等于阈值即为标记。"""
    record = _record(flagged=True, score=0.11)
    assert record.flagged is True
    with pytest.raises(StorageContractError):
        _record(flagged=False, score=0.11)


def test_indeterminate_record_keeps_the_explicit_model_flag() -> None:
    """规则不可评估时 ``signal_points`` 为 null，但 ``flagged`` 必须原样保留。

    这正是"从点数反推"会出错的分支：``indeterminate`` 没有点数，可是模型阈值结论
    （``score >= threshold``）与规则是否可评估无关。
    """
    assessment = build_assessment(flagged=True, matched=1, complete=False)
    assert assessment.signal_points is None
    assert assessment.risk_level == "indeterminate"

    record = StoredDecision.from_assessment(
        assessment,
        flagged=True,
        decision_id=DECISION_ID_A,
        created_at=CREATED_AT,
        evaluation_request_id="req-1",
        client_item_id=None,
        model_version="m1",
        score=0.9,
        threshold=0.11,
        policy_version="cw006-risk-policy-v1",
        policy_sha256="a" * 64,
    )
    assert record.flagged is True
    assert record.rule_evaluation_complete is False
    assert record.signal_points is None
    assert record.risk_level == "indeterminate"
    assert StoredDecision.from_row(record.to_row()) == record

    # 反例：同一个不可评估结果，显式传入未标记时也必须是未标记（不虚构标记）。
    unmarked = replace(record, flagged=False, score=0.05)
    assert unmarked.flagged is False


def test_read_row_rejects_a_flagged_contradiction() -> None:
    """读回同样强制该不变量：被篡改的行必须折叠为契约违规，而不是原样返回。"""
    tampered = _record(flagged=True).to_row()
    tampered["flagged"] = 0
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(tampered)

    raised_score = _record(flagged=False).to_row()
    raised_score["score"] = 0.90
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(raised_score)

    lowered_threshold = _record(flagged=False).to_row()
    lowered_threshold["threshold"] = 0.01
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(lowered_threshold)


def test_signal_points_out_of_range_is_rejected() -> None:
    row = _record(flagged=True).to_row()
    row["signal_points"] = 9
    with pytest.raises(StorageContractError):
        StoredDecision.from_row(row)


def test_all_risk_levels_are_recognised() -> None:
    assert set(ALL_RISK_LEVELS) == {
        "low_signal",
        "elevated_signal",
        "high_signal",
        "indeterminate",
    }
