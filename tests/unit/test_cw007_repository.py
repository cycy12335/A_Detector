"""CW-007 SQLite repository 的单元测试：schema 冻结、失败关闭、事务与参数化 SQL。

最要紧的四条：

1. **首次建库**写出冻结 schema：``user_version=1``、元数据表记录 ``cw007_storage_v1``、
   列集合完全等于允许字段；
2. **已有库严格校验**：版本/元数据/表/列/约束任一被篡改 → 失败关闭，且**不删除、不覆盖、
   不静默迁移**（文件字节不变）；
3. **写入是显式事务**：任一 insert/commit 失败 → 全部回滚，无部分记录；
4. **SQL 参数化**：带引号、SQL 关键字或超长 decision ID 的查询只得到"查不到"，不改变 schema。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from ai_detector.storage.config import (
    BUSY_TIMEOUT_MS,
    STORAGE_SCHEMA_VERSION,
    USER_VERSION,
    StorageErrorCode,
    StorageLoadError,
)
from ai_detector.storage.records import StoredDecision
from ai_detector.storage.sqlite import (
    DECISION_COLUMNS,
    SqliteDecisionRepository,
    StorageUnavailableError,
    open_repository,
)
from tests.cw007_fixtures import (
    CREATED_AT,
    DECISION_ID_A,
    DECISION_ID_B,
    DECISION_ID_C,
    build_assessment,
    remove_database,
    sqlite_connection,
)

ALLOWED_COLUMNS = set(DECISION_COLUMNS)


def _record(
    decision_id: str = DECISION_ID_A, *, matched: int = 1, flagged: bool = False, **overrides: object
) -> StoredDecision:
    """按 ``(flagged, matched)`` 一致地构造记录（跨字段不变量因此天然成立）。

    ``score`` 由 ``flagged`` 推导，因为 ``flagged == (score >= threshold)`` 是写入侧也要
    满足的跨字段不变量（阈值固定 ``0.11``）。
    """
    assessment = build_assessment(flagged=flagged, matched=matched)
    base = StoredDecision.from_assessment(
        assessment,
        flagged=flagged,
        decision_id=decision_id,
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


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """任意临时路径：直接测试 repository 时不需要走配置的目录限定。"""
    path = tmp_path / "decisions.sqlite3"
    yield path
    remove_database(path)


@pytest.fixture
def repository(db_path: Path) -> SqliteDecisionRepository:
    repo = open_repository(db_path)
    yield repo
    repo.close()


# --- 首次建库 -----------------------------------------------------------


def test_open_creates_the_frozen_schema(db_path: Path) -> None:
    assert not db_path.exists()
    repo = open_repository(db_path)
    try:
        assert db_path.is_file()
        with sqlite_connection(db_path) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == USER_VERSION
            assert (
                conn.execute(
                    "SELECT value FROM storage_metadata WHERE key = 'schema_version'"
                ).fetchone()[0]
                == STORAGE_SCHEMA_VERSION
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
            assert columns == ALLOWED_COLUMNS
    finally:
        repo.close()


def test_schema_has_no_feature_or_payload_columns(db_path: Path) -> None:
    repo = open_repository(db_path)
    try:
        with sqlite_connection(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        for banned in ("features", "feature_names", "payload", "body", "headers", "path", "query"):
            assert banned not in columns, banned
    finally:
        repo.close()


def test_schema_enforces_boolean_and_range_constraints(db_path: Path) -> None:
    """约束写在 schema 里：非 0/1 布尔与越界 rule_score 由 SQLite 自己拒绝。"""
    repo = open_repository(db_path)
    try:
        with sqlite_connection(db_path) as conn:
            ddl = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='decisions'"
            ).fetchone()[0]
            normalized = " ".join(ddl.split())
            assert "CHECK (flagged IN (0,1))" in normalized
            assert "CHECK (rule_evaluation_complete IN (0,1))" in normalized
            assert "rule_score >= 0.0" in normalized
            assert "decision_id TEXT PRIMARY KEY" in normalized

            def _row(**overrides: object) -> tuple:
                row = _record().to_row()
                row.update(overrides)
                return tuple(row[column] for column in DECISION_COLUMNS)

            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    f"INSERT INTO decisions ({','.join(DECISION_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(DECISION_COLUMNS))})",
                    _row(flagged=2),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    f"INSERT INTO decisions ({','.join(DECISION_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(DECISION_COLUMNS))})",
                    _row(rule_score=1.5),
                )
    finally:
        repo.close()


def test_open_is_idempotent_and_reuses_an_existing_database(db_path: Path) -> None:
    first = open_repository(db_path)
    try:
        first.insert_many([_record()])
    finally:
        first.close()

    second = open_repository(db_path)
    try:
        stored = second.get(DECISION_ID_A)
        assert stored is not None
        assert stored.decision_id == DECISION_ID_A
    finally:
        second.close()


def test_open_creates_the_parent_directory(db_path: Path) -> None:
    nested = db_path.parent / "nested" / "decisions.sqlite3"
    repo = open_repository(nested)
    try:
        assert nested.is_file()
    finally:
        repo.close()
        remove_database(nested)


# --- 已有库：失败关闭，不覆盖不迁移 ------------------------------------


def _tamper(db_path: Path, sql: str) -> None:
    with sqlite_connection(db_path) as conn:
        conn.execute(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "PRAGMA user_version = 3",   # 比当前 v2 更高：不得跳迁或猜测
        "PRAGMA user_version = 0",   # 未知版本：失败关闭
    ],
)
def test_wrong_user_version_fails_closed(db_path: Path, sql: str) -> None:
    repo = open_repository(db_path)
    repo.close()
    _tamper(db_path, sql)
    before = db_path.read_bytes()
    with pytest.raises(StorageLoadError) as error:
        open_repository(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert db_path.read_bytes() == before, "失败关闭时不得改写数据库文件"


def test_wrong_metadata_schema_version_fails_closed(db_path: Path) -> None:
    repo = open_repository(db_path)
    repo.close()
    _tamper(db_path, "UPDATE storage_metadata SET value='cw007_storage_v2' WHERE key='schema_version'")
    before = db_path.read_bytes()
    with pytest.raises(StorageLoadError) as error:
        open_repository(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert db_path.read_bytes() == before


def test_missing_metadata_row_fails_closed(db_path: Path) -> None:
    repo = open_repository(db_path)
    repo.close()
    _tamper(db_path, "DELETE FROM storage_metadata")
    with pytest.raises(StorageLoadError) as error:
        open_repository(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH


def test_extra_column_fails_closed(db_path: Path) -> None:
    repo = open_repository(db_path)
    repo.close()
    _tamper(db_path, "ALTER TABLE decisions ADD COLUMN extra_features TEXT")
    before = db_path.read_bytes()
    with pytest.raises(StorageLoadError) as error:
        open_repository(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert db_path.read_bytes() == before


def test_missing_table_fails_closed(db_path: Path) -> None:
    repo = open_repository(db_path)
    repo.close()
    _tamper(db_path, "DROP TABLE decisions")
    with pytest.raises(StorageLoadError) as error:
        open_repository(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH


def test_relaxed_check_constraint_fails_closed(db_path: Path) -> None:
    """把 CHECK 约束拿掉的"看起来一样"的表必须被识破。"""
    repo = open_repository(db_path)
    repo.close()
    _tamper(
        db_path,
        "CREATE TABLE decisions_relaxed AS SELECT * FROM decisions",
    )
    _tamper(db_path, "DROP TABLE decisions")
    _tamper(db_path, "ALTER TABLE decisions_relaxed RENAME TO decisions")
    before = db_path.read_bytes()
    with pytest.raises(StorageLoadError) as error:
        open_repository(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert db_path.read_bytes() == before


def test_non_sqlite_file_fails_closed(db_path: Path) -> None:
    db_path.write_bytes(b"this is definitely not a sqlite database" * 10)
    before = db_path.read_bytes()
    with pytest.raises(StorageLoadError) as error:
        open_repository(db_path)
    assert error.value.code in {StorageErrorCode.CORRUPT, StorageErrorCode.SCHEMA_MISMATCH}
    assert db_path.read_bytes() == before


# --- 已有库：结构漂移必须失败关闭（不只是"必需表存在"） -----------------


def _reopen_must_fail(db_path: Path, *statements: str) -> None:
    """篡改已有库后重开必须失败关闭，且**文件字节不变**（不覆盖、不迁移）。"""
    repo = open_repository(db_path)
    repo.close()
    for statement in statements:
        _tamper(db_path, statement)
    before = db_path.read_bytes()
    with pytest.raises(StorageLoadError) as error:
        open_repository(db_path)
    assert error.value.code == StorageErrorCode.SCHEMA_MISMATCH
    assert db_path.read_bytes() == before, "失败关闭时不得改写数据库文件"


def test_extra_table_fails_closed(db_path: Path) -> None:
    """只多一张"看起来无关"的表也不行：用户表集合必须精确等于冻结的两张。"""
    _reopen_must_fail(
        db_path, "CREATE TABLE unexpected_table (id INTEGER PRIMARY KEY, payload TEXT)"
    )


def test_extra_view_fails_closed(db_path: Path) -> None:
    _reopen_must_fail(
        db_path, "CREATE VIEW unexpected_view AS SELECT decision_id FROM decisions"
    )


def test_extra_trigger_fails_closed(db_path: Path) -> None:
    _reopen_must_fail(
        db_path,
        "CREATE TRIGGER unexpected_trigger AFTER INSERT ON decisions "
        "BEGIN UPDATE storage_metadata SET value = value; END",
    )


def test_extra_index_fails_closed(db_path: Path) -> None:
    """手工索引是结构漂移；SQLite 为 PK/UNIQUE 自建的 ``sqlite_autoindex_*`` 仍被放行。"""
    _reopen_must_fail(db_path, "CREATE INDEX unexpected_index ON decisions (score)")


def test_extra_metadata_row_fails_closed(db_path: Path) -> None:
    """元数据只允许冻结的 ``schema_version`` 一条记录。"""
    _reopen_must_fail(
        db_path,
        "INSERT INTO storage_metadata (key, value) VALUES ('extra_key', 'extra_value')",
    )


def test_extra_metadata_but_frozen_row_kept_fails_closed(db_path: Path) -> None:
    """即使冻结记录仍在，元数据表多出一列也必须让加载失败关闭。"""
    _reopen_must_fail(
        db_path,
        "CREATE TABLE metadata_wide (key TEXT PRIMARY KEY, value TEXT NOT NULL, note TEXT)",
        "INSERT INTO metadata_wide (key, value, note)"
        " SELECT key, value, 'x' FROM storage_metadata",
        "DROP TABLE storage_metadata",
        "ALTER TABLE metadata_wide RENAME TO storage_metadata",
    )


def test_metadata_table_structure_change_fails_closed(db_path: Path) -> None:
    """元数据表被换成缺少主键/约束的"同名表"必须被识破。"""
    _reopen_must_fail(
        db_path,
        "CREATE TABLE metadata_relaxed (key TEXT, value TEXT)",
        "INSERT INTO metadata_relaxed SELECT key, value FROM storage_metadata",
        "DROP TABLE storage_metadata",
        "ALTER TABLE metadata_relaxed RENAME TO storage_metadata",
    )


def test_decisions_not_null_dropped_fails_closed(db_path: Path) -> None:
    """列属性漂移：``CREATE TABLE ... AS SELECT`` 丢掉了 NOT NULL 与主键。"""
    _reopen_must_fail(
        db_path,
        "CREATE TABLE decisions_relaxed AS SELECT * FROM decisions",
        "DROP TABLE decisions",
        "ALTER TABLE decisions_relaxed RENAME TO decisions",
    )


def test_decisions_column_type_change_fails_closed(db_path: Path) -> None:
    """列类型漂移：``score`` 由 REAL 变成 TEXT 必须被识破。"""
    _reopen_must_fail(
        db_path,
        "CREATE TABLE decisions_retyped ("
        "decision_id TEXT PRIMARY KEY, record_schema_version TEXT NOT NULL,"
        "created_at TEXT NOT NULL, evaluation_request_id TEXT NOT NULL,"
        "client_item_id TEXT, model_version TEXT NOT NULL, score TEXT NOT NULL,"
        "score_semantics TEXT NOT NULL, threshold REAL NOT NULL,"
        "threshold_policy TEXT NOT NULL,"
        "flagged INTEGER NOT NULL CHECK (flagged IN (0,1)), policy_version TEXT NOT NULL,"
        "policy_sha256 TEXT NOT NULL, policy_semantics TEXT NOT NULL,"
        "rule_score REAL NOT NULL CHECK (rule_score >= 0.0 AND rule_score <= 1.0),"
        "matched_rule_ids TEXT NOT NULL, unevaluable_rule_ids TEXT NOT NULL,"
        "rule_evaluation_complete INTEGER NOT NULL"
        " CHECK (rule_evaluation_complete IN (0,1)), signal_points INTEGER,"
        "risk_level TEXT NOT NULL)",
        "INSERT INTO decisions_retyped SELECT * FROM decisions",
        "DROP TABLE decisions",
        "ALTER TABLE decisions_retyped RENAME TO decisions",
    )


def test_decisions_primary_key_moved_fails_closed(db_path: Path) -> None:
    """主键换到别的列上：列名与类型都对，但主键属性不同，必须被识破。"""
    _reopen_must_fail(
        db_path,
        "CREATE TABLE decisions_rekeyed ("
        "decision_id TEXT, record_schema_version TEXT NOT NULL,"
        "created_at TEXT NOT NULL, evaluation_request_id TEXT PRIMARY KEY,"
        "client_item_id TEXT, model_version TEXT NOT NULL, score REAL NOT NULL,"
        "score_semantics TEXT NOT NULL, threshold REAL NOT NULL,"
        "threshold_policy TEXT NOT NULL,"
        "flagged INTEGER NOT NULL CHECK (flagged IN (0,1)), policy_version TEXT NOT NULL,"
        "policy_sha256 TEXT NOT NULL, policy_semantics TEXT NOT NULL,"
        "rule_score REAL NOT NULL CHECK (rule_score >= 0.0 AND rule_score <= 1.0),"
        "matched_rule_ids TEXT NOT NULL, unevaluable_rule_ids TEXT NOT NULL,"
        "rule_evaluation_complete INTEGER NOT NULL"
        " CHECK (rule_evaluation_complete IN (0,1)), signal_points INTEGER,"
        "risk_level TEXT NOT NULL)",
        "INSERT INTO decisions_rekeyed SELECT * FROM decisions",
        "DROP TABLE decisions",
        "ALTER TABLE decisions_rekeyed RENAME TO decisions",
    )


# --- 已有库：约束必须**完全相等**，多一个也不行 ------------------------


def _with_extra_table_constraint(ddl: str, constraint: str) -> str:
    """在冻结 DDL 的最后一个右括号前插入一个**额外**表级约束。

    其余文本逐字保留（列名、顺序、类型、NOT NULL、主键与原有 CHECK 一个不动），因此重建出来的
    表只在"多了一个约束"这一点上与规范不同——这正是"包含式校验"会放行的那一类漂移。
    """
    head, _, _ = ddl.rpartition(")")
    return f"{head},\n    {constraint}\n)"


def test_decisions_extra_check_constraint_fails_closed(db_path: Path) -> None:
    """保留全部原有结构、只**增加**一个 CHECK 的 decisions 表必须被识破。"""
    from ai_detector.storage.sqlite import CANONICAL_DECISIONS_DDL

    _reopen_must_fail(
        db_path,
        "DROP TABLE decisions",
        _with_extra_table_constraint(CANONICAL_DECISIONS_DDL, "CHECK (score >= -999)"),
    )


def test_metadata_extra_check_constraint_fails_closed(db_path: Path) -> None:
    """元数据表同样要按完整 DDL 比对：多一个 CHECK 也必须失败关闭。

    重建后把冻结的 ``schema_version`` 记录**放回去**，因此本用例失败的原因只可能是多出来的
    约束，而不是"元数据行不见了"。
    """
    from ai_detector.storage.sqlite import CANONICAL_METADATA_DDL

    _reopen_must_fail(
        db_path,
        "DROP TABLE storage_metadata",
        _with_extra_table_constraint(CANONICAL_METADATA_DDL, "CHECK (length(value) > 0)"),
        "INSERT INTO storage_metadata (key, value) VALUES"
        f" ('schema_version', '{STORAGE_SCHEMA_VERSION}')",
    )


def test_metadata_extra_table_level_unique_fails_closed(db_path: Path) -> None:
    """表级 ``UNIQUE`` 是另一种未声明的约束：列属性完全不变，只有 DDL 能看出来。"""
    from ai_detector.storage.sqlite import CANONICAL_METADATA_DDL

    _reopen_must_fail(
        db_path,
        "DROP TABLE storage_metadata",
        _with_extra_table_constraint(CANONICAL_METADATA_DDL, "UNIQUE (value)"),
        "INSERT INTO storage_metadata (key, value) VALUES"
        f" ('schema_version', '{STORAGE_SCHEMA_VERSION}')",
    )


def test_canonical_ddl_matches_the_live_database(db_path: Path) -> None:
    """冻结的规范 DDL 必须与真实建库后 SQLite 存储的 DDL 逐字（折叠空白后）相等。

    这条守的是"建的"与"校验的"出自同一份常量：若建表语句与比对基准各自漂移，所有 DDL 用例
    都会失去意义。
    """
    from ai_detector.storage.sqlite import (
        CANONICAL_DECISIONS_DDL,
        CANONICAL_METADATA_DDL,
        _normalize_ddl,
    )

    repo = open_repository(db_path)
    try:
        with sqlite_connection(db_path) as conn:
            stored = {
                str(row[0]): _normalize_ddl(row[1])
                for row in conn.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
                )
            }
    finally:
        repo.close()
    assert stored["decisions"] == _normalize_ddl(CANONICAL_DECISIONS_DDL)
    assert stored["storage_metadata"] == _normalize_ddl(CANONICAL_METADATA_DDL)


def test_frozen_schema_definitions_match_the_live_database(db_path: Path) -> None:
    """冻结的列属性表必须与真实建库结果逐项一致（防止两份定义各自漂移）。"""
    from ai_detector.storage.sqlite import DECISION_COLUMN_SPECS, METADATA_COLUMN_SPECS
    repo = open_repository(db_path)
    try:
        with sqlite_connection(db_path) as conn:
            live = {
                table: tuple(
                    (row[1], row[2], row[3], row[5])
                    for row in conn.execute(f"PRAGMA table_info({table})")
                )
                for table in ("decisions", "storage_metadata")
            }
    finally:
        repo.close()
    assert live["decisions"] == DECISION_COLUMN_SPECS
    assert live["storage_metadata"] == METADATA_COLUMN_SPECS
    assert tuple(spec[0] for spec in DECISION_COLUMN_SPECS) == DECISION_COLUMNS


# --- 写入：显式事务与全成全败 ------------------------------------------


def test_insert_many_writes_all_records_in_one_call(repository: SqliteDecisionRepository) -> None:
    repository.insert_many(
        [_record(DECISION_ID_A), _record(DECISION_ID_B, matched=2), _record(DECISION_ID_C)]
    )
    assert repository.get(DECISION_ID_A) is not None
    assert repository.get(DECISION_ID_B) is not None
    assert repository.get(DECISION_ID_C) is not None


def test_duplicate_decision_id_rolls_the_whole_batch_back(
    repository: SqliteDecisionRepository,
) -> None:
    """已有 A 时再写 [B, A]：主键冲突必须让 **B 也不落库**。"""
    repository.insert_many([_record(DECISION_ID_A)])
    with pytest.raises(StorageUnavailableError):
        repository.insert_many([_record(DECISION_ID_B), _record(DECISION_ID_A)])
    assert repository.get(DECISION_ID_B) is None, "冲突批次必须整体回滚"
    assert repository.get(DECISION_ID_A) is not None


class _FailingConnection:
    """包装真实连接，让 ``commit`` 或 ``executemany`` 抛错。

    ``sqlite3.Connection`` / ``sqlite3.Cursor`` 是 C 类型，**不可** monkeypatch 方法，
    因此改为在 repository 自己的 ``_connect`` 接缝上注入包装器——这同样只验证产品代码路径。
    """

    def __init__(self, connection, *, fail_commit: bool = False, fail_executemany: bool = False):
        self._connection = connection
        self._fail_commit = fail_commit
        self._fail_executemany = fail_executemany

    def __getattr__(self, name):  # noqa: ANN001, ANN204
        return getattr(self._connection, name)

    def commit(self):  # noqa: ANN201
        if self._fail_commit:
            raise sqlite3.OperationalError("disk I/O error")
        return self._connection.commit()

    def executemany(self, sql, params):  # noqa: ANN001, ANN201
        if self._fail_executemany:
            raise sqlite3.OperationalError("database is locked")
        return self._connection.executemany(sql, params)


def _install_failing_connection(monkeypatch, db_path: Path, **flags: bool) -> None:
    @contextmanager
    def _connect(self):  # noqa: ANN001, ANN202
        raw = sqlite3.connect(db_path, isolation_level=None)
        raw.row_factory = sqlite3.Row
        try:
            yield _FailingConnection(raw, **flags)
        finally:
            raw.close()

    monkeypatch.setattr(SqliteDecisionRepository, "_connect", _connect)


def test_commit_failure_rolls_back_everything(
    repository: SqliteDecisionRepository, db_path: Path, monkeypatch
) -> None:
    _install_failing_connection(monkeypatch, db_path, fail_commit=True)
    with pytest.raises(StorageUnavailableError):
        repository.insert_many([_record(DECISION_ID_A), _record(DECISION_ID_B)])
    monkeypatch.undo()
    assert repository.get(DECISION_ID_A) is None
    assert repository.get(DECISION_ID_B) is None


def test_insert_failure_rolls_back_everything(
    repository: SqliteDecisionRepository, db_path: Path, monkeypatch
) -> None:
    _install_failing_connection(monkeypatch, db_path, fail_executemany=True)
    with pytest.raises(StorageUnavailableError):
        repository.insert_many([_record(DECISION_ID_A)])
    monkeypatch.undo()
    assert repository.get(DECISION_ID_A) is None


def test_empty_batch_is_a_no_op(repository: SqliteDecisionRepository) -> None:
    repository.insert_many([])


def test_duplicate_ids_within_one_batch_are_rejected(
    repository: SqliteDecisionRepository,
) -> None:
    with pytest.raises(StorageUnavailableError):
        repository.insert_many([_record(DECISION_ID_A), _record(DECISION_ID_A)])
    assert repository.get(DECISION_ID_A) is None


def test_inserted_row_round_trips_field_by_field(repository: SqliteDecisionRepository) -> None:
    record = _record(DECISION_ID_A, matched=2, flagged=True)
    repository.insert_many([record])
    stored = repository.get(DECISION_ID_A)
    assert stored == record


# --- 读取：参数化与安全 -------------------------------------------------


def test_get_returns_none_for_unknown_id(repository: SqliteDecisionRepository) -> None:
    assert repository.get(DECISION_ID_A) is None


@pytest.mark.parametrize(
    "hostile",
    [
        "' OR 1=1 --",
        "'; DROP TABLE decisions; --",
        '" OR ""="',
        "1; DELETE FROM decisions",
        "x' UNION SELECT * FROM storage_metadata --",
        "a" * 5000,
        "",
        "11111111-1111-4111-8111-111111111111' --",
    ],
)
def test_hostile_decision_ids_are_safe(
    repository: SqliteDecisionRepository, hostile: str
) -> None:
    """注入形状的 ID 只能"查不到"，不得改变 schema、抛错或返回别的记录。"""
    repository.insert_many([_record(DECISION_ID_A)])
    assert repository.get(hostile) is None
    # 表和元数据仍然完好
    assert repository.get(DECISION_ID_A) is not None


def test_corrupt_row_is_rejected_on_read(db_path: Path) -> None:
    """语义损坏的行必须被读取端拒绝。

    刻意改 ``risk_level`` 与 ``signal_points``：这两列**没有** CHECK 约束，因此损坏确实
    能落进表里——正是需要读取端重新校验的场景（``flagged`` 那类列由 CHECK 先挡住）。
    """
    repo = open_repository(db_path)
    try:
        repo.insert_many([_record(DECISION_ID_A)])
    finally:
        repo.close()
    with sqlite_connection(db_path) as conn:
        conn.execute(
            "UPDATE decisions SET risk_level = 'tampered' WHERE decision_id = ?",
            (DECISION_ID_A,),
        )
    reader = open_repository(db_path)
    try:
        with pytest.raises(StorageLoadError) as error:
            reader.get(DECISION_ID_A)
        assert error.value.code == StorageErrorCode.CONTRACT_VIOLATION
        assert "tampered" not in error.value.detail, "错误说明不得回显损坏行内容"
    finally:
        reader.close()


def test_row_with_non_boolean_flagged_cannot_even_be_written(db_path: Path) -> None:
    """第一道防线是 schema 里的 CHECK：非 0/1 布尔根本写不进表。"""
    repo = open_repository(db_path)
    try:
        repo.insert_many([_record(DECISION_ID_A)])
    finally:
        repo.close()
    with sqlite_connection(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE decisions SET flagged = 7 WHERE decision_id = ?", (DECISION_ID_A,))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE decisions SET rule_score = 2.0 WHERE decision_id = ?", (DECISION_ID_A,)
            )


def test_corrupt_json_row_is_rejected_on_read(db_path: Path) -> None:
    repo = open_repository(db_path)
    try:
        repo.insert_many([_record(DECISION_ID_A)])
    finally:
        repo.close()
    with sqlite_connection(db_path) as conn:
        conn.execute(
            "UPDATE decisions SET matched_rule_ids = 'not json' WHERE decision_id = ?",
            (DECISION_ID_A,),
        )
    reader = open_repository(db_path)
    try:
        with pytest.raises(StorageLoadError) as error:
            reader.get(DECISION_ID_A)
        assert error.value.code == StorageErrorCode.CONTRACT_VIOLATION
    finally:
        reader.close()


# --- 连接与 PRAGMA ------------------------------------------------------


def test_connections_are_short_lived_and_closed(db_path: Path) -> None:
    """每次操作自建连接并关闭，不保留全局可变连接。"""
    repo = open_repository(db_path)
    try:
        repo.insert_many([_record()])
        repo.get(DECISION_ID_A)
        # 没有未关闭连接时，Windows 上可以顺利删除文件
        repo.close()
        remove_database(db_path)
        assert not db_path.exists()
    finally:
        repo.close()


def test_pragmas_are_applied_per_connection(db_path: Path) -> None:
    repo = open_repository(db_path)
    try:
        with repo._connect() as conn:  # noqa: SLF001 - 定向验证 PRAGMA 设置
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
            assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "delete"
            assert str(conn.execute("PRAGMA synchronous").fetchone()[0]).lower() in {"2", "full"}
    finally:
        repo.close()


def test_foreign_keys_are_enforced_on_every_connection(db_path: Path) -> None:
    repo = open_repository(db_path)
    try:
        with repo._connect() as conn:  # noqa: SLF001
            conn.execute("CREATE TABLE IF NOT EXISTS probe_parent (id INTEGER PRIMARY KEY)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS probe_child ("
                "id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES probe_parent(id))"
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO probe_child (id, parent_id) VALUES (1, 999)")
    finally:
        repo.close()


# --- 文件内容扫描：不含载荷 --------------------------------------------


def test_database_file_contains_no_payload_or_feature_names(db_path: Path) -> None:
    repo = open_repository(db_path)
    try:
        repo.insert_many([_record(DECISION_ID_A, matched=2, flagged=True)])
    finally:
        repo.close()
    blob = db_path.read_bytes()
    for banned in (b"TransactionAmt", b"C13", b"features", b"smoke_constant", b"pickle"):
        assert banned not in blob, banned
    # 只允许确定性 JSON 列表，不允许 pickle 协议标记
    assert b"\x80\x04" not in blob


def test_stored_rule_ids_are_json_lists(db_path: Path) -> None:
    repo = open_repository(db_path)
    try:
        repo.insert_many([_record(DECISION_ID_A, matched=2)])
    finally:
        repo.close()
    with sqlite_connection(db_path) as conn:
        matched, unevaluable = conn.execute(
            "SELECT matched_rule_ids, unevaluable_rule_ids FROM decisions"
        ).fetchone()
    assert json.loads(matched) == [
        "transaction_amt_train_q99_tail",
        "c13_train_q99_tail",
    ]
    assert json.loads(unevaluable) == []
