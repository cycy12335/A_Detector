"""CW-008 测试夹具：严格 v1 数据库构造、记录构造、幂等键与指纹常量。

**为什么要在测试里手写 v1 建库**：CW-008 之后 repository 只会创建 v2，而"能不能迁移"必须用
**真真正正的 v1 库**来验证——不是把 v2 删两张表假装成 v1。这里用 CW-007 冻结的规范 DDL 与
列定义原样建库，因此迁移测试面对的就是生产上会遇到的那种 v1 文件。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Sequence

from ai_detector.storage.config import (
    LEGACY_STORAGE_SCHEMA_VERSION,
    LEGACY_USER_VERSION,
)
from ai_detector.storage.idempotency import (
    IDEMPOTENCY_SCHEMA_VERSION,
    OPERATION_SINGLE,
    key_digest,
    request_fingerprint,
    single_payload,
)
from ai_detector.storage.records import StoredDecision
from ai_detector.storage.sqlite import (
    CANONICAL_DECISIONS_DDL,
    CANONICAL_METADATA_DDL,
    DECISION_COLUMNS,
)
from tests.cw007_fixtures import CREATED_AT, build_assessment

#: 一组合法幂等键（各不相同，长度都 ≥ 16）。
KEY_ONE = "cw008-key-000000000001"
KEY_TWO = "cw008-key-000000000002"
KEY_THREE = "cw008-key-000000000003"


def build_record(
    decision_id: str,
    *,
    flagged: bool = False,
    matched: int = 1,
    client_item_id: str | None = None,
    created_at: str = CREATED_AT,
) -> StoredDecision:
    """按 ``(flagged, matched)`` 一致地构造记录（沿用 CW-007 的跨字段不变量）。"""
    assessment = build_assessment(flagged=flagged, matched=matched)
    return StoredDecision.from_assessment(
        assessment,
        flagged=flagged,
        decision_id=decision_id,
        created_at=created_at,
        evaluation_request_id="req-cw008",
        client_item_id=client_item_id,
        model_version="cw004-lightgbm-challenger-v1",
        score=0.9 if flagged else 0.05,
        threshold=0.11,
        policy_version="cw006-risk-policy-v1",
        policy_sha256="a" * 64,
    )


def connect(path: Path) -> sqlite3.Connection:
    """打开一个调用方负责关闭的连接（``with sqlite3.connect`` 不会关闭连接）。"""
    # ``isolation_level=None`` = 自动提交：夹具里每条语句自成一步，反例用例不会因为
    # 上一条失败语句把事务留在打开状态而互相污染。
    connection = sqlite3.connect(path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def snapshot_decisions(path: Path) -> list[tuple]:
    """按冻结列顺序、``decision_id`` 排序的规范行快照（用于迁移前后逐字段核对）。"""
    connection = connect(path)
    try:
        return [
            tuple(row[column] for column in DECISION_COLUMNS)
            for row in connection.execute(
                f"SELECT {', '.join(DECISION_COLUMNS)} FROM decisions ORDER BY decision_id"
            )
        ]
    finally:
        connection.close()


def make_v1_database(path: Path, records: Sequence[StoredDecision] = ()) -> Path:
    """在 ``path`` 建一个**严格合法**的 ``cw007_storage_v1`` 库（可选写入若干记录）。

    用的是 CW-007 冻结的规范 DDL 与元数据取值，因此它必须能通过 v1 档案的完整校验——
    这正是"迁移前先完整预校验"要抓到的东西。
    """
    connection = connect(path)
    try:
        connection.execute(CANONICAL_METADATA_DDL)
        connection.execute(CANONICAL_DECISIONS_DDL)
        connection.execute(
            "INSERT INTO storage_metadata (key, value) VALUES (?, ?)",
            ("schema_version", LEGACY_STORAGE_SCHEMA_VERSION),
        )
        if records:
            connection.executemany(
                f"INSERT INTO decisions ({', '.join(DECISION_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(DECISION_COLUMNS))})",
                [
                    tuple(record.to_row()[column] for column in DECISION_COLUMNS)
                    for record in records
                ],
            )
        connection.execute(f"PRAGMA user_version = {LEGACY_USER_VERSION}")
        connection.commit()
    finally:
        connection.close()
    return path


def tamper(path: Path, *statements: str) -> None:
    """直接改库（用于构造结构漂移、损坏行等反例）。

    刻意先关掉外键与 CHECK 约束：**反例要构造的正是"数据库本来不允许的状态"**。这些 pragma
    是连接级的，仓库自己打开的连接仍然照常启用约束，所以这里不会削弱生产路径。
    """
    connection = connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        for statement in statements:
            if statement.strip().upper().startswith("PRAGMA FOREIGN_KEYS"):
                continue
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def read_all(path: Path, table: str, columns: Sequence[str], order: str = "") -> list[tuple]:
    connection = connect(path)
    try:
        clause = f" ORDER BY {order}" if order else ""
        return [
            tuple(row[column] for column in columns)
            for row in connection.execute(f"SELECT {', '.join(columns)} FROM {table}{clause}")
        ]
    finally:
        connection.close()


def user_version(path: Path) -> int:
    connection = connect(path)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


def schema_objects(path: Path) -> list[tuple[str, str]]:
    connection = connect(path)
    try:
        return sorted(
            (str(row[0]), str(row[1]))
            for row in connection.execute("SELECT type, name FROM sqlite_master")
        )
    finally:
        connection.close()


def integrity_ok(path: Path) -> bool:
    connection = connect(path)
    try:
        return str(connection.execute("PRAGMA integrity_check").fetchone()[0]) == "ok"
    finally:
        connection.close()


def single_key_and_fingerprint(features: dict[str, Any], key: str = KEY_ONE) -> tuple[str, str, dict]:
    """返回 ``(key_digest, request_fingerprint, payload)``，供 repository 层测试直接使用。"""
    payload = single_payload(features)
    return (
        key_digest(key),
        request_fingerprint(OPERATION_SINGLE, payload),
        payload,
    )


__all__ = [
    "IDEMPOTENCY_SCHEMA_VERSION",
    "KEY_ONE",
    "KEY_THREE",
    "KEY_TWO",
    "build_record",
    "connect",
    "integrity_ok",
    "make_v1_database",
    "read_all",
    "schema_objects",
    "single_key_and_fingerprint",
    "snapshot_decisions",
    "tamper",
    "user_version",
]
