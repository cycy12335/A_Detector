"""CW-005 请求契约的单元测试（纯 Pydantic，不走 HTTP）。

三条严格性各自独立可测：envelope 不许额外字段、取值只允许四种类型、批量 ID 必须唯一。
其中"拒绝 bool"最容易被忽略——Python 里 ``bool`` 是 ``int`` 的子类，若不做显式判断，
``True`` 会被静默当成 ``1``，类型契约就形同虚设。
"""

from __future__ import annotations

import pytest

# 服务专项依赖来自 `service` extra：未安装时清晰跳过本模块，安装后必须正常执行（不得跳过）。
pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from pydantic import ValidationError

from ai_detector.service.schemas import (
    BatchScoreRequest,
    ScoreItem,
    ScoreRequest,
    check_feature_value,
)


# --- 取值类型 -----------------------------------------------------------


@pytest.mark.parametrize("value", ["text", "", 0, 1, -7, 0.0, 1.5, -0.25, None, 10**18])
def test_allowed_value_types_are_accepted(value: object) -> None:
    assert check_feature_value(value, where="probe") == value


@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_rejected(value: bool) -> None:
    """bool 是 int 的子类：不显式拒绝就会被静默当成 0/1。"""
    with pytest.raises(ValueError, match="不允许布尔值"):
        check_feature_value(value, where="probe")


@pytest.mark.parametrize("value", [[1, 2], {"nested": 1}, (1, 2), {1, 2}])
def test_nested_containers_are_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="只允许"):
        check_feature_value(value, where="probe")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="NaN 或 Infinity"):
        check_feature_value(value, where="probe")


def test_rejection_message_names_the_offending_feature() -> None:
    with pytest.raises(ValueError, match=r"features\[amt\]"):
        check_feature_value(True, where="features[amt]")


# --- envelope 严格性 ----------------------------------------------------


def test_single_request_accepts_a_plain_features_map() -> None:
    request = ScoreRequest(features={"a": 1.0, "b": "x", "c": None})
    assert request.features == {"a": 1.0, "b": "x", "c": None}


def test_single_request_rejects_extra_envelope_field() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest(features={"a": 1.0}, unexpected="nope")


def test_single_request_requires_features() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest()


def test_single_request_rejects_bool_value_through_the_model() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest(features={"a": True})


def test_batch_request_accepts_items() -> None:
    request = BatchScoreRequest(
        items=[
            ScoreItem(client_item_id="a", features={"x": 1.0}),
            ScoreItem(client_item_id="b", features={"x": 2.0}),
        ]
    )
    assert [item.client_item_id for item in request.items] == ["a", "b"]


def test_batch_request_rejects_extra_envelope_field() -> None:
    with pytest.raises(ValidationError):
        BatchScoreRequest(items=[{"client_item_id": "a", "features": {}}], limit=5)


def test_batch_request_rejects_item_extra_field() -> None:
    with pytest.raises(ValidationError):
        BatchScoreRequest(items=[{"client_item_id": "a", "features": {}, "extra": 1}])


def test_batch_request_rejects_empty_items() -> None:
    with pytest.raises(ValidationError):
        BatchScoreRequest(items=[])


def test_batch_request_rejects_duplicate_client_item_ids() -> None:
    with pytest.raises(ValidationError, match="client_item_id 必须唯一"):
        BatchScoreRequest(
            items=[
                {"client_item_id": "dup", "features": {"x": 1.0}},
                {"client_item_id": "dup", "features": {"x": 2.0}},
            ]
        )


def test_batch_request_does_not_echo_duplicate_ids_in_the_error() -> None:
    """错误信息只说数量，不回显调用方提供的 ID。"""
    with pytest.raises(ValidationError) as error:
        BatchScoreRequest(
            items=[
                {"client_item_id": "SENTINEL_ITEM_ID", "features": {}},
                {"client_item_id": "SENTINEL_ITEM_ID", "features": {}},
            ]
        )
    assert "SENTINEL_ITEM_ID" not in str(error.value)


def test_batch_request_rejects_blank_client_item_id() -> None:
    with pytest.raises(ValidationError):
        BatchScoreRequest(items=[{"client_item_id": "   ", "features": {}}])


def test_client_item_id_length_is_bounded() -> None:
    with pytest.raises(ValidationError):
        BatchScoreRequest(items=[{"client_item_id": "x" * 129, "features": {}}])


def test_no_probability_or_disposition_field_names_exist() -> None:
    """响应模型不得出现概率或**处置**类字段名。

    CW-007 新增了一个名字里含 ``decision`` 的字段：``DecisionResponse.decision`` 承载的是
    "已持久化的决策**记录**"（一个数据容器），不是批准/拒绝这类处置结论。因此这里把"处置
    语义"的判定收紧成逐字段的**语义**约束，并**额外**禁止 ``block`` / ``allow``，
    而不是对 ``decision`` 一词做全局封杀——那会把"记录容器"和"生产处置"混为一谈。

    改动记录见 ``WORK_LOG.md`` 的 CW-007 小节：这是本包唯一一处调整既有断言的测试，
    调整后覆盖**更多**名字（新增 block/allow），并对 ``decision`` 追加了类型断言。
    """
    from ai_detector.service import schemas
    from ai_detector.service.schemas import DecisionRecord

    banned = ("probability", "fraud_probability", "approve", "decline", "block", "allow")
    for name in dir(schemas):
        model = getattr(schemas, name)
        fields = getattr(model, "model_fields", None)
        if not fields:
            continue
        for field_name, field in fields.items():
            assert field_name.lower() not in banned, f"{name}.{field_name}"
            if field_name == "decision":
                # 唯一允许的 `decision` 字段必须是记录容器，而不是处置结论。
                assert name == "DecisionResponse", f"{name}.decision"
                assert field.annotation is DecisionRecord, f"{name}.decision 必须是记录容器"
