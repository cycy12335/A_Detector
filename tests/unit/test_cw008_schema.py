"""CW-008 SQLite v2 schema 与**唯一允许的 v1→v2 迁移**的单元测试。

最要紧的五条：

1. 新文件直接建出**严格 v2**（四张表、``user_version=2``、元数据 ``cw008_storage_v2``、
   DDL/列/约束逐项精确）；
2. 完整合法的 v1 被**原子**升级，既有决策记录逐字段不变；
3. 重复启动**零写入**；
4. 任何结构漂移、未知版本、损坏记录都失败关闭，且**文件字节不变**；
5. 迁移中途失败只回滚自己那个事务，不留"半迁移"。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ai_detector.storage.config import (
    LEGACY_STORAGE_SCHEMA_VERSION,
    LEGACY_USER_VERSION,
    STORAGE_SCHEMA_VERSION,
    USER_VERSION,
    StorageErrorCode,
    StorageLoadError,
)
from ai_detector.storage.sqlite import (
    CANONICAL_IDEMPOTENCY_REQUESTS_DDL,
    CANONICAL_IDEMPOTENCY_RESULTS_DDL,
    DECISION_COLUMNS,
    IDEMPOTENCY_REQUEST_COLUMN_SPECS,
    IDEMPOTENCY_RESULT_COLUMN_SPECS,
    _create_statement,
    _normalize_ddl,
    open_repository,
)
from tests.cw007_fixtures import remove_database
from tests.cw008_fixtures import (
    build_record,
    connect,
    integrity_ok,
    make_v1_database,
    read_all,
    schema_objects,
    snapshot_decisions,
    tamper,
    user_version,
)

#: 迁移测试用的一组规范 UUID4（数量覆盖 0/1/5 行三种情形）。
V1_IDS = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
    "44444444-4444-4444-8444-444444444444",
    "55555555-5555-4555-8555-555555555555",
)

EXPECTED_TABLES = {
    "decisions",
    "storage_metadata",
    "idempotency_requests",
    "idempotency_results",
}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "cw008.sqlite3"
    yield path
    remove_database(path)


def _open(path: Path):
    repository = open_repository(path)
    return repository


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 新库：严格 v2 ------------------------------------------------------


def test_new_database_is_strict_v2(db_path: Path) -> None:
    repository = _open(db_path)
    repository.close()
    assert user_version(db_path) == USER_VERSION == 2
    assert {name for kind, name in schema_objects(db_path) if kind == "table"} == EXPECTED_TABLES


def test_new_database_metadata_and_columns_are_frozen(db_path: Path) -> None:
    repository = _open(db_path)
    repository.close()
    connection = connect(db_path)
    try:
        metadata = [
            (str(row[0]), str(row[1])) for row in connection.execute("SELECT key, value FROM storage_metadata")
        ]
        assert metadata == [("schema_version", STORAGE_SCHEMA_VERSION)]

        requests_specs = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute("PRAGMA table_info(idempotency_requests)")
        )
        results_specs = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute("PRAGMA table_info(idempotency_results)")
        )
        stored_ddl = {
            str(row[0]): _normalize_ddl(row[1])
            for row in connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()
    assert requests_specs == IDEMPOTENCY_REQUEST_COLUMN_SPECS
    assert results_specs == IDEMPOTENCY_RESULT_COLUMN_SPECS
    assert stored_ddl["idempotency_requests"] == _normalize_ddl(CANONICAL_IDEMPOTENCY_REQUESTS_DDL)
    assert stored_ddl["idempotency_results"] == _normalize_ddl(CANONICAL_IDEMPOTENCY_RESULTS_DDL)


def test_idempotency_tables_enforce_shape_at_the_database_level(db_path: Path) -> None:
    """摘要长度、operation 域、result_count 范围与索引范围都由 CHECK 兜底。"""
    import sqlite3

    repository = _open(db_path)
    repository.close()
    connection = connect(db_path)
    try:
        def _insert_request(**overrides: object) -> None:
            row = {
                "key_digest": "a" * 64,
                "idempotency_schema_version": "cw008_idempotency_record_v1",
                "operation": "decision_single",
                "fingerprint_version": "cw008_request_fingerprint_v1",
                "request_fingerprint": "b" * 64,
                "result_count": 1,
            }
            row.update(overrides)
            connection.execute(
                "INSERT INTO idempotency_requests VALUES (?, ?, ?, ?, ?, ?)",
                tuple(row.values()),
            )

        _insert_request()
        # 注意：SQLite 有类型亲和性，'1' 这类可无损转换的取值会被存成整数 1，因此这里只列
        # **结构上真的越界**的取值；非整数 result_count 由读取侧的契约校验兜底。
        for bad in (
            {"operation": "decision_bulk"},
            {"request_fingerprint": "b" * 63},
            {"result_count": 0},
            {"result_count": 101},
        ):
            with pytest.raises(sqlite3.IntegrityError):
                _insert_request(key_digest="c" * 64, **bad)
        with pytest.raises(sqlite3.IntegrityError):   # 非 0/1 之外的越界索引
            connection.execute(
                "INSERT INTO idempotency_results VALUES (?, ?, ?)", ("a" * 64, 99, "x")
            )
    finally:
        connection.close()


def test_foreign_keys_are_enforced_between_new_tables(db_path: Path) -> None:
    """结果行必须指向真实存在的请求行与决策行。"""
    import sqlite3

    repository = _open(db_path)
    repository.close()
    connection = connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO idempotency_results VALUES (?, ?, ?)",
                ("a" * 64, 0, "11111111-1111-4111-8111-111111111111"),
            )
    finally:
        connection.close()


# --- v1 -> v2 迁移 ------------------------------------------------------


def test_v1_database_with_no_rows_migrates(db_path: Path) -> None:
    make_v1_database(db_path)
    repository = _open(db_path)
    repository.close()
    assert user_version(db_path) == USER_VERSION
    assert {name for kind, name in schema_objects(db_path) if kind == "table"} == EXPECTED_TABLES
    connection = connect(db_path)
    try:
        assert [
            (str(row[0]), str(row[1])) for row in connection.execute("SELECT key, value FROM storage_metadata")
        ] == [("schema_version", STORAGE_SCHEMA_VERSION)]
    finally:
        connection.close()


@pytest.mark.parametrize("count", [1, 5])
def test_v1_database_rows_survive_migration_field_by_field(db_path: Path, count: int) -> None:
    ids = V1_IDS[:count]
    records = [build_record(decision_id, flagged=index % 2 == 0, client_item_id=f"item-{index}")
               for index, decision_id in enumerate(ids)]
    make_v1_database(db_path, records)
    before = snapshot_decisions(db_path)
    assert len(before) == count

    repository = _open(db_path)
    repository.close()

    assert user_version(db_path) == USER_VERSION
    assert snapshot_decisions(db_path) == before, "迁移必须逐字段保留既有决策记录"
    assert integrity_ok(db_path)
    assert read_all(db_path, "idempotency_requests", ("key_digest",)) == []
    assert read_all(db_path, "idempotency_results", ("key_digest",)) == []


def test_reopening_a_migrated_database_performs_zero_writes(db_path: Path) -> None:
    make_v1_database(db_path, [build_record(V1_IDS[0])])
    first = _open(db_path)
    first.close()
    migrated = _digest(db_path)
    for _ in range(3):
        repository = _open(db_path)
        repository.close()
        assert _digest(db_path) == migrated, "重复启动不得再次迁移或改写文件"


def test_v1_with_unknown_shape_fails_closed_and_does_not_migrate(db_path: Path) -> None:
    make_v1_database(db_path)
    tamper(db_path, "CREATE TABLE unexpected_table (id INTEGER PRIMARY KEY)")
    before = _digest(db_path)
    with pytest.raises(StorageLoadError) as error:
        _open(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert _digest(db_path) == before, "预校验失败时文件必须一字不变"
    assert user_version(db_path) == LEGACY_USER_VERSION


def test_v1_with_extra_check_constraint_fails_closed(db_path: Path) -> None:
    """v1 也必须**结构完全相等**：多一个 CHECK 就不是"完整合法 v1"。"""
    from ai_detector.storage.sqlite import CANONICAL_DECISIONS_DDL

    make_v1_database(db_path)
    head, _, _ = CANONICAL_DECISIONS_DDL.rpartition(")")
    tamper(
        db_path,
        "DROP TABLE decisions",
        f"{head},\n    CHECK (score >= -999)\n)",
    )
    before = _digest(db_path)
    with pytest.raises(StorageLoadError) as error:
        _open(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert _digest(db_path) == before


@pytest.mark.parametrize("version", [0, 3, 99])
def test_unknown_user_version_fails_closed(db_path: Path, version: int) -> None:
    make_v1_database(db_path)
    tamper(db_path, f"PRAGMA user_version = {version}")
    before = _digest(db_path)
    with pytest.raises(StorageLoadError) as error:
        _open(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert _digest(db_path) == before


def test_v1_with_corrupt_record_fails_closed(db_path: Path) -> None:
    """记录读不出来就不算"完整合法 v1"：结构完全相等也救不了坏行，不得带着坏行进 v2。"""
    make_v1_database(db_path, [build_record(V1_IDS[0])])
    tamper(db_path, "UPDATE decisions SET risk_level = 'not_a_frozen_level'")
    before = _digest(db_path)
    with pytest.raises(StorageLoadError) as error:
        _open(db_path)
    assert error.value.code == StorageErrorCode.CONTRACT_VIOLATION
    assert _digest(db_path) == before
    assert user_version(db_path) == LEGACY_USER_VERSION


def test_metadata_version_mismatch_in_v1_fails_closed(db_path: Path) -> None:
    make_v1_database(db_path)
    tamper(db_path, "UPDATE storage_metadata SET value = 'cw009_storage_v3'")
    before = _digest(db_path)
    with pytest.raises(StorageLoadError) as error:
        _open(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert _digest(db_path) == before


def test_migration_failure_rolls_back_and_leaves_v1_intact(db_path: Path, monkeypatch) -> None:
    """注入事务中途失败：只回滚本次事务，v1 schema、记录与 integrity 全部不变，无半迁移。"""
    from ai_detector.storage import sqlite as sqlite_module

    records = [build_record(V1_IDS[index]) for index in range(3)]
    make_v1_database(db_path, records)
    before = snapshot_decisions(db_path)

    monkeypatch.setattr(
        sqlite_module,
        "_CREATE_IDEMPOTENCY_RESULTS_SQL",
        "CREATE TABLE idempotency_results (this is not valid sql",
    )
    with pytest.raises(StorageLoadError):
        _open(db_path)

    assert user_version(db_path) == LEGACY_USER_VERSION, "失败后不得留下已提升的版本号"
    assert snapshot_decisions(db_path) == before
    assert integrity_ok(db_path)
    tables = {name for kind, name in schema_objects(db_path) if kind == "table"}
    assert tables == {"decisions", "storage_metadata"}, "不得留下半迁移的新表"
    connection = connect(db_path)
    try:
        value = connection.execute(
            "SELECT value FROM storage_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert value == LEGACY_STORAGE_SCHEMA_VERSION


def test_non_sqlite_file_fails_closed(db_path: Path) -> None:
    db_path.write_bytes(b"definitely not a sqlite database" * 20)
    before = _digest(db_path)
    with pytest.raises(StorageLoadError) as error:
        _open(db_path)
    assert error.value.code in {StorageErrorCode.CORRUPT, StorageErrorCode.SCHEMA_MISMATCH}
    assert _digest(db_path) == before


def test_v2_with_extra_table_fails_closed(db_path: Path) -> None:
    repository = _open(db_path)
    repository.close()
    tamper(db_path, "CREATE TABLE unexpected_table (id INTEGER PRIMARY KEY)")
    before = _digest(db_path)
    with pytest.raises(StorageLoadError) as error:
        _open(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert _digest(db_path) == before


def test_create_statement_of_new_tables_is_derived_from_the_canonical_ddl() -> None:
    """建表语句必须由冻结 DDL 派生（同一份文本），否则"建的"与"校验的"会各自漂移。"""
    assert _create_statement(CANONICAL_IDEMPOTENCY_REQUESTS_DDL).startswith(
        "CREATE TABLE IF NOT EXISTS idempotency_requests"
    )
    assert "CREATE TABLE IF NOT EXISTS idempotency_results" in _create_statement(
        CANONICAL_IDEMPOTENCY_RESULTS_DDL
    )


def test_decisions_table_is_untouched_by_the_upgrade(db_path: Path) -> None:
    """CW-008 不得改写 ``decisions``：列集合与记录 schema 逐字不变。"""
    from ai_detector.storage.sqlite import DECISION_COLUMN_SPECS

    make_v1_database(db_path, [build_record(V1_IDS[0])])
    repository = _open(db_path)
    repository.close()
    connection = connect(db_path)
    try:
        specs = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute("PRAGMA table_info(decisions)")
        )
        record_schema = connection.execute("SELECT record_schema_version FROM decisions").fetchone()[0]
    finally:
        connection.close()
    assert specs == DECISION_COLUMN_SPECS
    assert tuple(spec[0] for spec in specs) == DECISION_COLUMNS
    assert record_schema == "cw007_decision_record_v1"
