"""CW-008 幂等键校验与规范请求指纹（纯函数，无 I/O）。

## 为什么"指纹"要写死一套规范化规则

幂等的全部语义都压在一个判断上：**这两个请求是不是同一个请求**。如果这个判断依赖字典
插入顺序、``1`` 与 ``1.0`` 的写法或 ``-0.0`` 的符号，那么同一个业务请求换一种等价写法就会被
判成"不同请求"，从而产生重复记录——幂等就名存实亡。因此这里把规范化规则固化成代码：

- 对象 key 按 Unicode code point 排序，JSON 用 UTF-8、``ensure_ascii=False``、无多余空白；
- features 的输入顺序不影响指纹；批量 item 顺序、``client_item_id``、字符串与 ``null`` 保持原语义；
- 有限数统一转 float（``1`` ≡ ``1.0``），``-0.0`` 规范为 ``0.0``；``True``/``False`` 是**布尔**，
  绝不与 ``1``/``0`` 归一；NaN/Infinity 直接拒绝；
- ``operation`` 进入指纹，单笔与批量因此不会共享命名空间。

## 它不是什么

产出的两个摘要都只是 **SHA-256 摘要**：它们会泄漏"两个输入是否相等"，不是匿名化、不是加密，
也不是不可逆的安全凭据。原始 key、规范 JSON、features 与请求体**一律不入库、不入日志**。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

#: 幂等记录 schema 版本（写入 ``idempotency_requests``）。
IDEMPOTENCY_SCHEMA_VERSION = "cw008_idempotency_record_v1"

#: 请求指纹算法版本（写入 ``idempotency_requests``）。
FINGERPRINT_VERSION = "cw008_request_fingerprint_v1"

#: 两个 operation 域：单笔与批量各自独立，防止同 payload 跨端点误重放。
OPERATION_SINGLE = "decision_single"
OPERATION_BATCH = "decision_batch"
OPERATIONS: tuple[str, ...] = (OPERATION_SINGLE, OPERATION_BATCH)

#: 幂等 header 名与响应头名。
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
REPLAYED_HEADER = "Idempotency-Replayed"

#: 合法 key 的长度边界（含）与冻结正则。取值大小写敏感，不 trim、不做 Unicode 归一化。
IDEMPOTENCY_KEY_MIN_LENGTH = 16
IDEMPOTENCY_KEY_MAX_LENGTH = 128
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")

#: 批量上限（与 1–100 的既有契约一致；仅用于 ``result_count`` 的范围校验）。
MAX_RESULT_COUNT = 100


class IdempotencyKeyError(ValueError):
    """幂等键形状非法。

    **不得**把原值放进消息：这个错误会流向 400 响应与日志白名单之外的地方。
    """


class IdempotencyContractError(ValueError):
    """已持久化的幂等映射或关联损坏。

    与 :class:`IdempotencyKeyError` 分开，是因为二者对外的错误分类完全不同：
    前者是调用方的 400，后者是**服务端存储**的 500。消息同样不得回显任何摘要或 ID。
    """


#: 摘要形状：64 位小写十六进制。
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def is_digest(value: Any) -> bool:
    """是否是规范的 64 位小写十六进制摘要。"""
    return isinstance(value, str) and bool(_DIGEST_PATTERN.fullmatch(value))


def validate_idempotency_key(value: Any) -> str:
    """校验并**原样**返回幂等键。

    Args:
        value: 待校验的 header 取值。

    Returns:
        与输入逐字相同的字符串（不 trim、不改大小写、不做归一化）。

    Raises:
        IdempotencyKeyError: 非字符串、长度不在 16–128、含非 ASCII 或非法字符、
            前后有空白。消息只说明违规类别，不回显取值。
    """
    if not isinstance(value, str):
        raise IdempotencyKeyError("幂等键必须是字符串")
    if not IDEMPOTENCY_KEY_PATTERN.fullmatch(value):
        raise IdempotencyKeyError(
            "幂等键必须是 16-128 个 ASCII 字符，且以字母或数字开头，"
            "只允许字母、数字与 . _ : -"
        )
    return value


def key_digest(key: str) -> str:
    """对原始 key 的 UTF-8 字节计算小写 SHA-256（唯一入库的 key 相关值）。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def require_operation(operation: str) -> str:
    """确认 operation 属于冻结的两个域之一。"""
    if operation not in OPERATIONS:
        raise ValueError("operation 必须是冻结的两个域之一")
    return operation


def _canonicalize(value: Any) -> Any:
    """把已校验的请求取值递归规范化成确定性的中间表示。

    只接受 JSON 取值（``null`` / 布尔 / 数 / 字符串 / 数组 / 对象）；其它类型一律拒绝，
    以免"某天多出一种类型"时静默退化成 ``str(value)`` 这种不稳定的表示。
    """
    if isinstance(value, bool):  # 必须在 int 之前：bool 是 int 的子类
        return value
    if isinstance(value, int | float):
        try:
            number = float(value)
        except OverflowError as error:  # pragma: no cover - 极端大整数
            raise ValueError("数值超出可表示范围") from error
        if not math.isfinite(number):
            raise ValueError("请求中不得出现 NaN 或 Infinity")
        # ``-0.0 == 0.0`` 为真，因此这一步同时消除负零。
        return 0.0 if number == 0.0 else number
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    raise ValueError("请求中出现不支持的取值类型")


def canonical_request_json(operation: str, payload: Mapping[str, Any]) -> str:
    """返回规范 JSON 文本（**只用于计算指纹**，绝不落库或写日志）。"""
    require_operation(operation)
    document = {"operation": operation, "payload": _canonicalize(payload)}
    return json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def request_fingerprint(operation: str, payload: Mapping[str, Any]) -> str:
    """对规范 JSON 的 UTF-8 字节计算小写 SHA-256。"""
    return hashlib.sha256(
        canonical_request_json(operation, payload).encode("utf-8")
    ).hexdigest()


def single_payload(features: Mapping[str, Any]) -> dict[str, Any]:
    """单笔请求的规范载荷（只取已校验的 ``features``）。"""
    return {"features": dict(features)}


def batch_payload(items: Any) -> dict[str, Any]:
    """批量请求的规范载荷：**保持请求顺序**并保留 ``client_item_id``。

    顺序与 ``client_item_id`` 都进入指纹，因此"同一批打乱顺序"或"换一个关联 ID"都算不同请求。
    """
    return {
        "items": [
            {"client_item_id": str(item.client_item_id), "features": dict(item.features)}
            for item in items
        ]
    }
