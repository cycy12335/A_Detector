"""CW-009 报告落盘：原子写入 + 落盘前的脱敏断言。

## 报告是"可定位"的，不是"可复现输入"的

报告只写：工作包、schema、UTC 时间、阶段状态与稳定错误码、四个实际摘要、版本、
case/覆盖/失败计数、失败 case ID 与字段名、score 最大绝对偏差、pytest 命令的安全形状与
计数、环境版本。它**不**写 feature 取值、完整 feature 映射、请求体、数据库行、UUID、
幂等摘要、绝对路径、bundle 内容、pytest 节点 ID 或捕获的输出。

这条约束不靠"写代码时小心"，而是靠 :func:`assert_report_safe` 在落盘前再查一遍——
查出违规就**失败关闭**（退出 6），宁可不产出报告，也不产出一份泄漏的报告。

## 原子写入

先写同目录下的临时文件，``flush`` + ``fsync`` 后用 :func:`os.replace` 原子替换。
任何一步失败都会删掉临时文件，不留半写报告。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode

#: schema 版本（写在报告里，供下游判定字段含义）。
REPORT_SCHEMA_VERSION = "cw009_quality_gate_report_v1"

_SHA256_TOKEN = re.compile(r"\b[0-9a-f]{64}\b")
_UUID_TOKEN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_UNC_PREFIX = "\\\\"


def _iter_strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield from _iter_strings(key)
            yield from _iter_strings(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_strings(item)


def _looks_absolute(value: str) -> bool:
    """绝对路径判定（平台无关）：盘符、UNC、扩展长度前缀、POSIX 根路径。"""
    if not value:
        return False
    if _WINDOWS_DRIVE.match(value):
        return True
    if value.startswith(_UNC_PREFIX):
        return True
    return value.startswith("/")


def assert_report_safe(
    report: Mapping[str, Any],
    *,
    allowed_digests: Iterable[str],
    forbidden_tokens: Iterable[str],
) -> None:
    """落盘前的脱敏断言。

    Args:
        report: 待写入的报告对象。
        allowed_digests: 允许出现的 64 位摘要（冻结值与实际值）。
        forbidden_tokens: 绝不允许出现的字符串（源特征名与类别合成取值）。

    Raises:
        QualityGateError: 出现绝对路径、UUID、计划外摘要或禁用字符串
            （``report_unsafe``）。
    """
    allowed = {str(item).lower() for item in allowed_digests}
    forbidden = {str(item) for item in forbidden_tokens}
    for text in _iter_strings(report):
        if _looks_absolute(text):
            raise QualityGateError(QualityGateErrorCode.REPORT_UNSAFE, "报告含绝对路径")
        if _UUID_TOKEN.search(text):
            raise QualityGateError(QualityGateErrorCode.REPORT_UNSAFE, "报告含 UUID")
        for token in _SHA256_TOKEN.findall(text.lower()):
            if token not in allowed:
                raise QualityGateError(
                    QualityGateErrorCode.REPORT_UNSAFE, "报告含计划外的摘要"
                )
        for token in text.split():
            if token in forbidden:
                raise QualityGateError(
                    QualityGateErrorCode.REPORT_UNSAFE, "报告含源特征名或类别取值"
                )


def dumps(report: Mapping[str, Any]) -> str:
    """稳定的 JSON 文本（UTF-8、缩进 2、保留非 ASCII）。"""
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def write_report(
    path: Path,
    report: Mapping[str, Any],
    *,
    allowed_digests: Iterable[str],
    forbidden_tokens: Iterable[str],
) -> None:
    """先脱敏断言，再原子写入。

    Raises:
        QualityGateError: 脱敏断言失败（``report_unsafe``）或写入失败
            （``report_write_failed``）。
    """
    assert_report_safe(
        report, allowed_digests=allowed_digests, forbidden_tokens=forbidden_tokens
    )
    payload = dumps(report).encode("utf-8")

    handle = None
    temporary_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        handle = os.fdopen(descriptor, "wb")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temporary_name, path)
        temporary_name = None
    except OSError as error:
        raise QualityGateError(
            QualityGateErrorCode.REPORT_WRITE_FAILED,
            f"报告写入失败：{type(error).__name__}",
        ) from error
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:  # pragma: no cover - 关闭失败时已无补救手段
                pass
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:  # pragma: no cover - 临时文件已被移走或无法删除
                pass
