"""CW-007 最小化决策记录持久化（标准库 ``sqlite3``）。

本包把 CW-006 的研究型决策结果**最小化**地写入本地 SQLite，并支持按服务端 UUID4
``decision_id`` 原样读回。它不保存任何输入载荷（features、请求体、header、原始 URL），
也不实现幂等、列表查询、更新或删除。

模块划分：

- :mod:`ai_detector.storage.config` —— 冻结版本常量与安全的项目相对数据库路径；
- :mod:`ai_detector.storage.records` —— 不可变记录及其严格编码/解码；
- :mod:`ai_detector.storage.sqlite` —— schema 初始化、参数化读写与显式事务。

**它不是什么**：不是数据仓库，不做迁移、复制、加密或跨进程扩展；数据库是**可变运行时
产物**，可信度来自固定 schema、约束与读取时的严格校验，而不是指纹。
"""

from __future__ import annotations

from ai_detector.storage.config import (
    BUSY_TIMEOUT_MS,
    DEFAULT_STORAGE_PATH,
    LEGACY_STORAGE_SCHEMA_VERSION,
    LEGACY_USER_VERSION,
    RECORD_SCHEMA_VERSION,
    STORAGE_DIR,
    STORAGE_SCHEMA_VERSION,
    STORAGE_SUFFIX,
    USER_VERSION,
    StorageErrorCode,
    StorageLoadError,
    assert_safe_storage_path,
    resolve_storage_path,
)
from ai_detector.storage.idempotency import (
    FINGERPRINT_VERSION,
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENCY_KEY_PATTERN,
    IDEMPOTENCY_SCHEMA_VERSION,
    OPERATION_BATCH,
    OPERATION_SINGLE,
    OPERATIONS,
    REPLAYED_HEADER,
    IdempotencyContractError,
    IdempotencyKeyError,
    batch_payload,
    canonical_request_json,
    key_digest,
    request_fingerprint,
    single_payload,
    validate_idempotency_key,
)
from ai_detector.storage.records import StoredDecision, StorageContractError
from ai_detector.storage.sqlite import (
    CURRENT_PROFILE,
    DECISION_COLUMNS,
    IDEMPOTENCY_CREATED,
    IDEMPOTENCY_MISS,
    IDEMPOTENCY_REPLAYED,
    LEGACY_PROFILE,
    DecisionRepository,
    IdempotencyHit,
    SchemaProfile,
    SqliteDecisionRepository,
    StorageUnavailableError,
    open_repository,
)

__all__ = [
    "BUSY_TIMEOUT_MS",
    "CURRENT_PROFILE",
    "DECISION_COLUMNS",
    "DEFAULT_STORAGE_PATH",
    "DecisionRepository",
    "FINGERPRINT_VERSION",
    "IDEMPOTENCY_CREATED",
    "IDEMPOTENCY_KEY_HEADER",
    "IDEMPOTENCY_KEY_PATTERN",
    "IDEMPOTENCY_MISS",
    "IDEMPOTENCY_REPLAYED",
    "IDEMPOTENCY_SCHEMA_VERSION",
    "IdempotencyContractError",
    "IdempotencyHit",
    "IdempotencyKeyError",
    "LEGACY_PROFILE",
    "LEGACY_STORAGE_SCHEMA_VERSION",
    "LEGACY_USER_VERSION",
    "OPERATIONS",
    "OPERATION_BATCH",
    "OPERATION_SINGLE",
    "RECORD_SCHEMA_VERSION",
    "REPLAYED_HEADER",
    "STORAGE_DIR",
    "STORAGE_SCHEMA_VERSION",
    "STORAGE_SUFFIX",
    "SchemaProfile",
    "SqliteDecisionRepository",
    "StorageContractError",
    "StorageErrorCode",
    "StorageLoadError",
    "StorageUnavailableError",
    "StoredDecision",
    "USER_VERSION",
    "assert_safe_storage_path",
    "batch_payload",
    "canonical_request_json",
    "key_digest",
    "open_repository",
    "request_fingerprint",
    "resolve_storage_path",
    "single_payload",
    "validate_idempotency_key",
]
