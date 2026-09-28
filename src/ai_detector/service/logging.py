"""无载荷的结构化日志：每个请求**恰好一条** JSON completion log。

## 记什么、不记什么

只记录"这次调用发生了什么"这一类**元数据**：

- ``request_id``、``method``、``route``（模板路径，不含具体 ID）、``status_code``、
  ``duration_ms``、``model_version``（可用时）、``batch_size``（适用时）、``outcome``。

**绝不记录**：请求/响应 body、features 取值、``client_item_id``、分数、header 集合、
文件路径、异常堆栈、凭据。这些要么是调用方的业务数据，要么是内部结构信息，写进日志
只会扩大泄漏面，而排障所需的关联信息一个 ``request_id`` 就够了。

## 为什么 message 就是 JSON 字符串

日志行本身就是完整对象，不需要额外的 formatter 解析；这样任何日志后端（stdout、文件、
采集器）拿到的都是可直接索引的一行 JSON。模块不主动添加 handler——由进程入口
（``scripts/run_api.py``）配置，避免在测试与库调用场景下产生重复输出。
"""

from __future__ import annotations

import json
import logging
from typing import Any

#: 服务统一使用的 logger 名。
LOGGER_NAME = "ai_detector.service"

#: completion log 中允许出现的字段白名单。
#: 用白名单而不是"记得别写"：任何新增字段都必须显式出现在这里，否则会被丢弃。
#:
#: 启动诊断复用同一份白名单，因此它同样**不可能**写出路径、配置载荷、异常文本或堆栈——
#: 那些字段根本不在这里。
ALLOWED_LOG_FIELDS: frozenset[str] = frozenset(
    {
        "event",
        "request_id",
        "method",
        "route",
        "status_code",
        "duration_ms",
        "model_version",
        "batch_size",
        "outcome",
        # 启动诊断专用（同样只允许枚举值与标识，不含路径或异常文本）
        "service",
        "api_version",
        "error_code",
        "n_source_features",
        # CW-006 / 政策相关：只允许**标识、计数与枚举等级**。
        # 规则 ID 列表、rule score、特征值与匹配明细一律不在白名单内。
        "policy_version",
        "policy_error_code",
        "risk_level",
        "rule_match_count",
        # CW-007 / 存储相关：只允许操作的**类别**与**计数**，以及固定 schema 版本。
        # decision_id、evaluation_request_id、client_item_id、数据库路径与 SQL 一律不在白名单内。
        "storage_operation",
        # CW-008：只记**结果枚举**（not_requested / created / replayed / conflict）。
        # 原始 key、key digest、request fingerprint 一律不在白名单内。
        "idempotency_outcome",
        "record_count",
        "storage_ready",
        "storage_schema_version",
        "storage_error_code",
    }
)

#: 启动诊断的事件名；与请求日志的 ``request_completed`` 区分开，
#: 因此"每请求恰好一条 completion log"的计数不受启动日志影响。
STARTUP_EVENT = "service_startup"

logger = logging.getLogger(LOGGER_NAME)


def scrub(fields: dict[str, Any]) -> dict[str, Any]:
    """按白名单过滤字段，并丢弃 ``None`` 值。

    白名单是最后一道防线：即便调用方不小心把 body 或 features 塞进 payload，
    也不会被写出去。
    """
    return {
        key: value
        for key, value in fields.items()
        if key in ALLOWED_LOG_FIELDS and value is not None
    }


def log_completion(fields: dict[str, Any]) -> dict[str, Any]:
    """写出一条 JSON completion log，返回实际写出的字典（便于测试断言）。"""
    payload = scrub(fields)
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))
    return payload


def log_startup(
    *,
    service: str,
    api_version: str,
    model_ready: bool,
    policy_ready: bool,
    model_version: str | None = None,
    n_source_features: int | None = None,
    model_error_code: str | None = None,
    policy_version: str | None = None,
    policy_error_code: str | None = None,
    storage_ready: bool = True,
    storage_schema_version: str | None = None,
    storage_error_code: str | None = None,
) -> dict[str, Any]:
    """写出一条结构化**启动诊断**，返回实际写出的字典。

    成功与失败各写一条，供运维从进程日志判断服务是否就绪。``outcome`` 为
    ``"ready"`` 仅当**模型、政策与存储都就绪**；任一失败即为 ``"not_ready"``。
    ``storage_ready`` 默认 ``True``，因此**未配置存储**的部署不会因此被判为未就绪。

    诊断**区分各组件**：就绪的组件报出版本/特征数/schema 版本，未就绪的组件只报**安全分类码**
    （如 ``bundle_hash_mismatch``、``policy_hash_mismatch``、``storage_schema_mismatch``），
    不含绝对路径、配置载荷、数据库内容、SQL、异常文本或堆栈——这些在进程里就地丢弃。

    它使用与请求日志相同的白名单，因此不可能写出上列敏感内容；
    事件名是 ``service_startup``，不会被误计入"每请求一条 completion log"。
    """
    fields: dict[str, Any] = {
        "event": STARTUP_EVENT,
        "service": service,
        "api_version": api_version,
        "outcome": "ready"
        if (model_ready and policy_ready and storage_ready)
        else "not_ready",
    }
    if model_ready:
        fields["model_version"] = model_version
        fields["n_source_features"] = n_source_features
    if policy_ready:
        fields["policy_version"] = policy_version
    if storage_ready:
        fields["storage_ready"] = True
        fields["storage_schema_version"] = storage_schema_version
    if not (model_ready and policy_ready and storage_ready):
        # 阻塞项按"模型 → 政策 → 存储"的顺序取第一个：越靠前的组件越贴近阻塞原因。
        blocker = model_error_code or policy_error_code or storage_error_code
        if blocker is not None:
            fields["error_code"] = blocker
        if policy_error_code is not None:
            fields["policy_error_code"] = policy_error_code
        if storage_error_code is not None:
            fields["storage_error_code"] = storage_error_code
    payload = scrub(fields)
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))
    return payload


def configure_logging(*, level: int = logging.INFO) -> logging.Logger:
    """为服务 logger 配置单行 JSON 输出（供进程入口调用）。

    设置 ``propagate = False`` 并对本 logger 去重 handler，因此一条请求只会输出一行，
    不会因为 root logger 再输出一次而变成两条。
    """
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(handler, "_ai_detector_service", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._ai_detector_service = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger
