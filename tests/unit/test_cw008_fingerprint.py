"""CW-008 幂等键与规范请求指纹的单元测试（纯函数，不碰数据库）。

三条最要紧的性质：

1. **key 契约严格**：16–128 个 ASCII 字符、大小写敏感、不 trim、不归一化；非法形状必须在
   任何模型/规则/数据库动作**之前**被拒。
2. **指纹确定**：与字典顺序无关、与等价数值写法无关（``1`` ≡ ``1.0``、``-0.0`` ≡ ``0.0``），
   但与 batch 顺序、``client_item_id``、operation 域**强相关**。
3. **只产出摘要**：原始 key 与规范 JSON 都不得出现在返回的摘要里。
"""

from __future__ import annotations

import hashlib

import pytest

from ai_detector.storage.idempotency import (
    FINGERPRINT_VERSION,
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENCY_KEY_MAX_LENGTH,
    IDEMPOTENCY_KEY_MIN_LENGTH,
    IDEMPOTENCY_SCHEMA_VERSION,
    OPERATION_BATCH,
    OPERATION_SINGLE,
    IdempotencyKeyError,
    canonical_request_json,
    key_digest,
    request_fingerprint,
    validate_idempotency_key,
)

#: 合法 key：正好 16 位与正好 128 位。
KEY_MIN = "a" * IDEMPOTENCY_KEY_MIN_LENGTH
KEY_MAX = "Z9._:-" * 21 + "ab"          # 126 + 2 = 128
KEY_TYPICAL = "11111111-2222-4333-8444-555555555555"


def _single(features: dict) -> dict:
    return {"operation": OPERATION_SINGLE, "payload": {"features": features}}


# --- 冻结常量 -----------------------------------------------------------


def test_frozen_versions_and_header_name() -> None:
    assert IDEMPOTENCY_SCHEMA_VERSION == "cw008_idempotency_record_v1"
    assert FINGERPRINT_VERSION == "cw008_request_fingerprint_v1"
    assert OPERATION_SINGLE == "decision_single"
    assert OPERATION_BATCH == "decision_batch"
    assert IDEMPOTENCY_KEY_HEADER == "Idempotency-Key"
    assert (IDEMPOTENCY_KEY_MIN_LENGTH, IDEMPOTENCY_KEY_MAX_LENGTH) == (16, 128)
    assert len(KEY_MAX) == 128, "夹具自身必须是边界值"


# --- key 语法 -----------------------------------------------------------


@pytest.mark.parametrize("value", [KEY_MIN, KEY_MAX, KEY_TYPICAL, "abcDEF012._:-xyz",
                                   "0" * 16, "x" * 128])
def test_valid_keys_are_accepted_verbatim(value: str) -> None:
    assert validate_idempotency_key(value) == value, "不得 trim 或改写取值"


@pytest.mark.parametrize(
    "value",
    [
        "",
        " " * 16,
        "a" * 15,                       # 过短
        "a" * 129,                      # 过长
        " " + "a" * 15,                 # 前导空格
        "a" * 15 + " ",                 # 尾随空格
        "-" + "a" * 15,                 # 首字符必须是字母数字
        "." + "a" * 15,
        ":" + "a" * 15,
        "a" * 15 + "!",                 # 非法字符
        "a" * 15 + "/",
        "a" * 15 + "+",
        "a" * 15 + "\t",
        "a" * 15 + "\n",
        "密" * 16,                       # 非 ASCII
        "é" + "a" * 15,
        "a" * 15 + "é",
        "a b" + "a" * 13,               # 内部空格
        "a" * 15 + " ",            # 不换行空格
    ],
)
def test_invalid_keys_are_rejected(value: str) -> None:
    with pytest.raises(IdempotencyKeyError):
        validate_idempotency_key(value)


@pytest.mark.parametrize("value", [None, 123, b"a" * 16, ["a" * 16], {"k": "v"}])
def test_non_string_keys_are_rejected(value: object) -> None:
    with pytest.raises(IdempotencyKeyError):
        validate_idempotency_key(value)


def test_key_error_message_never_echoes_the_value() -> None:
    secret = "SENTINEL_KEY_0123456789"
    with pytest.raises(IdempotencyKeyError) as error:
        validate_idempotency_key(secret + "!")
    assert secret not in str(error.value)


def test_keys_are_case_sensitive() -> None:
    assert key_digest(KEY_MIN.upper()) != key_digest(KEY_MIN.lower())


# --- key 摘要 -----------------------------------------------------------


def test_key_digest_is_lowercase_hex_sha256_of_the_key_bytes() -> None:
    expected = hashlib.sha256(KEY_TYPICAL.encode("utf-8")).hexdigest()
    digest = key_digest(KEY_TYPICAL)
    assert digest == expected
    assert len(digest) == 64 and digest == digest.lower()
    assert KEY_TYPICAL not in digest


def test_key_digest_distinguishes_neighbouring_keys() -> None:
    assert key_digest("a" * 16) != key_digest("a" * 15 + "b")


# --- 规范 JSON ----------------------------------------------------------


def test_canonical_json_is_compact_sorted_and_unicode_preserving() -> None:
    text = canonical_request_json(OPERATION_SINGLE, {"features": {"b": 1, "a": "值"}})
    assert text == '{"operation":"decision_single","payload":{"features":{"a":"值","b":1.0}}}'
    assert "\\u" not in text, "非 ASCII 必须原样保留（ensure_ascii=False）"


def test_canonical_json_has_no_trailing_or_leading_space() -> None:
    text = canonical_request_json(OPERATION_SINGLE, {"features": {}})
    assert text == text.strip()


# --- 指纹：顺序无关 -----------------------------------------------------


def test_feature_order_does_not_affect_the_fingerprint() -> None:
    a = request_fingerprint(OPERATION_SINGLE, {"features": {"x": 1.0, "y": 2.0}})
    b = request_fingerprint(OPERATION_SINGLE, {"features": {"y": 2.0, "x": 1.0}})
    assert a == b


def test_nested_object_order_does_not_affect_the_fingerprint() -> None:
    a = request_fingerprint(OPERATION_BATCH, {"items": [{"features": {"p": 1.0, "q": 2.0}}]})
    b = request_fingerprint(OPERATION_BATCH, {"items": [{"features": {"q": 2.0, "p": 1.0}}]})
    assert a == b


# --- 指纹：数值归一 -----------------------------------------------------


def test_int_and_float_forms_are_equivalent() -> None:
    assert request_fingerprint(OPERATION_SINGLE, {"features": {"x": 1}}) == request_fingerprint(
        OPERATION_SINGLE, {"features": {"x": 1.0}}
    )


def test_negative_zero_normalises_to_zero() -> None:
    assert request_fingerprint(OPERATION_SINGLE, {"features": {"x": -0.0}}) == request_fingerprint(
        OPERATION_SINGLE, {"features": {"x": 0.0}}
    )


def test_booleans_are_not_numeric() -> None:
    """``True`` 是合法取值，但绝不能被当成 ``1.0``。"""
    assert request_fingerprint(OPERATION_SINGLE, {"features": {"x": True}}) != request_fingerprint(
        OPERATION_SINGLE, {"features": {"x": 1.0}}
    )
    assert request_fingerprint(OPERATION_SINGLE, {"features": {"x": False}}) != request_fingerprint(
        OPERATION_SINGLE, {"features": {"x": 0.0}}
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        request_fingerprint(OPERATION_SINGLE, {"features": {"x": value}})
    with pytest.raises(ValueError):
        canonical_request_json(OPERATION_SINGLE, {"features": {"x": value}})


def test_null_and_string_values_survive_normalisation() -> None:
    a = request_fingerprint(OPERATION_SINGLE, {"features": {"x": None, "y": "0"}})
    b = request_fingerprint(OPERATION_SINGLE, {"features": {"x": None, "y": "0.0"}})
    assert a != b, "字符串必须是字符串，不能与数字归一"


# --- 指纹：语义敏感 -----------------------------------------------------


def test_batch_item_order_is_significant() -> None:
    first = {"items": [{"client_item_id": "a", "features": {"x": 1.0}},
                       {"client_item_id": "b", "features": {"x": 1.0}}]}
    swapped = {"items": [{"client_item_id": "b", "features": {"x": 1.0}},
                         {"client_item_id": "a", "features": {"x": 1.0}}]}
    assert request_fingerprint(OPERATION_BATCH, first) != request_fingerprint(OPERATION_BATCH, swapped)


def test_client_item_id_is_significant() -> None:
    a = {"items": [{"client_item_id": "a", "features": {}}]}
    b = {"items": [{"client_item_id": "b", "features": {}}]}
    assert request_fingerprint(OPERATION_BATCH, a) != request_fingerprint(OPERATION_BATCH, b)


def test_batch_size_is_significant() -> None:
    one = {"items": [{"client_item_id": "a", "features": {}}]}
    two = {"items": [{"client_item_id": "a", "features": {}}, {"client_item_id": "b", "features": {}}]}
    assert request_fingerprint(OPERATION_BATCH, one) != request_fingerprint(OPERATION_BATCH, two)


def test_operation_domain_separates_identical_payloads() -> None:
    """``operation`` 必须进入指纹，否则单笔与批量会共享命名空间。"""
    payload = {"features": {"x": 1.0}}
    assert request_fingerprint(OPERATION_SINGLE, payload) != request_fingerprint(
        OPERATION_BATCH, payload
    )


def test_fingerprint_is_lowercase_hex_sha256_and_deterministic() -> None:
    payload = {"features": {"x": 1.0}}
    first = request_fingerprint(OPERATION_SINGLE, payload)
    second = request_fingerprint(OPERATION_SINGLE, payload)
    assert first == second
    assert len(first) == 64 and first == first.lower()
    assert all(character in "0123456789abcdef" for character in first)


def test_fingerprint_does_not_contain_the_payload() -> None:
    sentinel = "SENTINEL_FEATURE_0123456789"
    digest = request_fingerprint(OPERATION_SINGLE, {"features": {"x": sentinel}})
    assert sentinel not in digest
    assert len(digest) == 64


def test_unknown_operation_is_rejected() -> None:
    with pytest.raises(ValueError):
        request_fingerprint("decision_bulk", {"features": {}})


def test_key_and_fingerprint_use_different_inputs() -> None:
    """key 摘要与请求指纹互不影响：同一 key 配不同请求必须得到不同指纹。"""
    key = KEY_TYPICAL
    a = request_fingerprint(OPERATION_SINGLE, {"features": {"x": 1.0}})
    b = request_fingerprint(OPERATION_SINGLE, {"features": {"x": 2.0}})
    assert key_digest(key) == key_digest(key)
    assert a != b
