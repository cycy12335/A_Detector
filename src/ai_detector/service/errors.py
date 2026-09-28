"""统一的安全错误结构。

对外只有一种错误形状::

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

## 为什么要自己接管错误

FastAPI / Pydantic 的默认校验错误体会把**原始 input** 回显出来（``detail[].input``）。
那意味着攻击者或误操作者提交的特征值、甚至被塞进请求的凭据片段，会原样出现在响应里；
再叠加上默认的堆栈与内部异常文本，错误响应就成了信息泄漏面。因此本模块：

- 覆盖默认校验处理器，只给**稳定的错误码 + 面向调用方的说明**；
- 任何情况下都不回显 input、特征值、header 集合、绝对路径、堆栈或内部异常文本；
- 未知异常统一折叠为 ``internal_error``，细节只进入服务端日志（且同样不含载荷）。
"""

from __future__ import annotations

from typing import Any

#: 稳定错误码 -> 面向调用方的安全说明。
ERROR_MESSAGES: dict[str, str] = {
    "invalid_request_id": "请求 ID 不合法：只允许 1-64 位 ASCII 字母、数字与 . _ : - 。",
    "invalid_request_body": "请求体不符合接口契约。",
    "feature_contract_violation": "features 必须与模型要求的源特征集合完全一致，且取值类型合法。",
    "batch_contract_violation": "批量请求不符合契约（数量、唯一性或结构）。",
    "model_not_ready": "模型当前不可用：bundle 未通过校验或尚未加载完成。",
    "policy_not_ready": "风险规则政策当前不可用：政策缺失、未通过校验或尚未加载完成。",
    "policy_contract_violation": "风险规则政策未能产生一致的结果，服务端政策需要修复。",
    "storage_not_ready": "决策存储当前不可用：未配置、未通过校验或尚未初始化完成。",
    "storage_unavailable": "决策存储暂时不可用，请稍后重试。",
    "storage_contract_violation": "存储中的决策记录结构不一致，服务端存储需要修复。",
    "decision_not_found": "指定的决策 ID 不存在。",
    # CW-008：文案固定，且**不含**任何 key、摘要或指纹。
    "invalid_idempotency_key": (
        "Idempotency-Key 不合法：必须是 16-128 个 ASCII 字符，以字母或数字开头，"
        "只允许字母、数字与 . _ : - ，且同一请求只能出现一次。"
    ),
    "idempotency_conflict": "该 Idempotency-Key 已用于另一个请求；复用同一 key 时请求内容必须一致。",
    "idempotency_contract_violation": "服务端幂等映射已损坏，需要人工修复后才能继续使用该接口。",
    "not_found": "请求的接口不存在。",
    "method_not_allowed": "该接口不支持此 HTTP 方法。",
    "internal_error": "服务内部错误，请稍后重试或联系维护者。",
}


class ServiceErrorCode:
    """错误码常量（避免在代码里散落字符串字面量）。"""

    INVALID_REQUEST_ID = "invalid_request_id"
    INVALID_REQUEST_BODY = "invalid_request_body"
    FEATURE_CONTRACT_VIOLATION = "feature_contract_violation"
    BATCH_CONTRACT_VIOLATION = "batch_contract_violation"
    MODEL_NOT_READY = "model_not_ready"
    POLICY_NOT_READY = "policy_not_ready"
    POLICY_CONTRACT_VIOLATION = "policy_contract_violation"
    STORAGE_NOT_READY = "storage_not_ready"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    STORAGE_CONTRACT_VIOLATION = "storage_contract_violation"
    DECISION_NOT_FOUND = "decision_not_found"
    INVALID_IDEMPOTENCY_KEY = "invalid_idempotency_key"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    IDEMPOTENCY_CONTRACT_VIOLATION = "idempotency_contract_violation"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    INTERNAL_ERROR = "internal_error"


class ApiError(Exception):
    """受控的对外错误。

    Args:
        status_code: HTTP 状态码。
        code: 稳定错误码。
        message: 面向调用方的说明；省略时使用 :data:`ERROR_MESSAGES` 的默认文案。
        internal_detail: **只写日志、不进响应**的补充说明（同样不得含载荷或绝对路径）。
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str | None = None,
        *,
        internal_detail: str | None = None,
    ) -> None:
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = message or ERROR_MESSAGES.get(self.code, "请求无法完成。")
        self.internal_detail = internal_detail
        super().__init__(f"{self.status_code} {self.code}")

    def to_body(self, request_id: str) -> dict[str, Any]:
        """构造对外的错误体（只含码、说明与请求 ID）。"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "request_id": request_id,
            }
        }


class BundleLoadError(RuntimeError):
    """bundle 无法安全加载。

    Raises 时**不得**附带绝对路径或文件内容片段；用 :attr:`code` 表达原因类别，
    细节由调用方以安全措辞记录。
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


class BundleErrorCode:
    """bundle 加载失败的原因分类（用于启动诊断）。"""

    MISSING = "bundle_missing"
    HASH_MISMATCH = "bundle_hash_mismatch"
    DESERIALIZE_FAILED = "bundle_deserialize_failed"
    WRONG_TYPE = "bundle_wrong_type"
    SCHEMA_MISMATCH = "bundle_schema_mismatch"
    FEATURE_CONTRACT = "bundle_feature_contract"
    THRESHOLD_INVALID = "bundle_threshold_invalid"
    CONTAINS_RAW_DATA = "bundle_contains_raw_data"
    DEPENDENCY_MISSING = "bundle_dependency_missing"
