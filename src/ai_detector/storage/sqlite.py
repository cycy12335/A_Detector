"""CW-007 SQLite repository：固定 schema、参数化读写、显式事务、连接及时关闭。

## 连接模型

**每次操作自建一个连接并在 ``finally`` 中关闭**，不保存任何全局可变连接，也不跨线程复用。
代价是每操作一次 connect（本地 SQLite 上是微秒级），换来的是"没有跨请求状态"这一确定性。

## 事务模型

写入使用 ``BEGIN IMMEDIATE`` … ``commit()`` 的显式事务：批量要么**全部**落库，要么**全部**
回滚。任何 ``sqlite3`` 异常都会先 ``rollback()`` 再折叠成稳定的 :class:`StorageUnavailableError`
——**不返回**"只写了前半批"的部分结果，也**不允许**回退成"只返回未持久化结果"。

## schema 纪律

首次打开空文件时创建固定 schema 并写入 ``user_version=1``；**已存在**的数据库必须逐项匹配
``user_version``、元数据版本、表、列与约束，任一不符即失败关闭——**不删除、不覆盖、不迁移**。
数据库是可变运行时产物，因此可信度来自"固定 schema + 约束 + 读取时重新校验"，
而不是指纹。

## SQL 纪律

所有语句都是模块常量 + ``?`` 占位符；唯一拼接进 SQL 的是冻结的列名元组与冻结的整数
``user_version``，**没有**任何调用方数据参与字符串拼接。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from ai_detector.storage.config import (
    BUSY_TIMEOUT_MS,
    FOREIGN_KEYS,
    JOURNAL_MODE,
    LEGACY_STORAGE_SCHEMA_VERSION,
    LEGACY_USER_VERSION,
    STORAGE_SCHEMA_VERSION,
    SYNCHRONOUS,
    USER_VERSION,
    StorageErrorCode,
    StorageLoadError,
)
from ai_detector.storage.idempotency import (
    FINGERPRINT_VERSION,
    IDEMPOTENCY_SCHEMA_VERSION,
    MAX_RESULT_COUNT,
    OPERATIONS,
    OPERATION_BATCH,
    OPERATION_SINGLE,
    IdempotencyContractError,
    is_digest,
)
from ai_detector.storage.records import (
    StorageContractError,
    StoredDecision,
    is_canonical_decision_id,
)

#: 两张幂等表的表名。
_IDEMPOTENCY_REQUESTS_TABLE = "idempotency_requests"
_IDEMPOTENCY_RESULTS_TABLE = "idempotency_results"

#: 决策表的列属性冻结表：``(名字, 类型, NOT NULL, 主键序号)``。
#:
#: 顺序即 schema 顺序；``DECISION_COLUMNS`` 由它派生，因此"写读用的列"与"校验用的列"
#: 不可能各写一份而漂移。注意 ``decision_id TEXT PRIMARY KEY`` 在 SQLite 里是
#: ``notnull=0, pk=1``（非 INTEGER 主键在 rowid 表上允许 NULL，这是 SQLite 的既有语义）。
DECISION_COLUMN_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("decision_id", "TEXT", 0, 1),
    ("record_schema_version", "TEXT", 1, 0),
    ("created_at", "TEXT", 1, 0),
    ("evaluation_request_id", "TEXT", 1, 0),
    ("client_item_id", "TEXT", 0, 0),
    ("model_version", "TEXT", 1, 0),
    ("score", "REAL", 1, 0),
    ("score_semantics", "TEXT", 1, 0),
    ("threshold", "REAL", 1, 0),
    ("threshold_policy", "TEXT", 1, 0),
    ("flagged", "INTEGER", 1, 0),
    ("policy_version", "TEXT", 1, 0),
    ("policy_sha256", "TEXT", 1, 0),
    ("policy_semantics", "TEXT", 1, 0),
    ("rule_score", "REAL", 1, 0),
    ("matched_rule_ids", "TEXT", 1, 0),
    ("unevaluable_rule_ids", "TEXT", 1, 0),
    ("rule_evaluation_complete", "INTEGER", 1, 0),
    ("signal_points", "INTEGER", 0, 0),
    ("risk_level", "TEXT", 1, 0),
)

#: 元数据表的列属性冻结表（同样按 ``(名字, 类型, NOT NULL, 主键序号)``）。
METADATA_COLUMN_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("key", "TEXT", 0, 1),
    ("value", "TEXT", 1, 0),
)

#: 决策表的列，顺序即 schema 顺序与读写顺序（由 :data:`DECISION_COLUMN_SPECS` 派生）。
DECISION_COLUMNS: tuple[str, ...] = tuple(spec[0] for spec in DECISION_COLUMN_SPECS)

_METADATA_TABLE = "storage_metadata"
_DECISIONS_TABLE = "decisions"

#: SQLite 为 PRIMARY KEY / UNIQUE 自动建的索引前缀；只有这些内部索引被放行。
_AUTO_INDEX_PREFIX = "sqlite_autoindex_"

#: 两张用户表的**规范 DDL**（冻结文本，公开常量）。
#:
#: 它们是**唯一**的 schema 真值来源，同时承担两个角色：
#:
#: - **建表**：:func:`_create_statement` 在它上面插入 ``IF NOT EXISTS`` 得到建表语句；
#: - **校验**：已有库的 ``sqlite_master.sql`` 经 :func:`_normalize_ddl` 折叠空白后与它
#:   **精确比较**（而不是搜索"必需片段"）。
#:
#: 文本刻意按 SQLite **实际存储**的形态书写——SQLite 存 DDL 时会去掉 ``IF NOT EXISTS``，
#: 所以这里也不带它。两者出自同一份常量，因此"建的"与"校验的"不可能各自漂移。
CANONICAL_DECISIONS_DDL = f"""
CREATE TABLE {_DECISIONS_TABLE} (
    decision_id TEXT PRIMARY KEY,
    record_schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    evaluation_request_id TEXT NOT NULL,
    client_item_id TEXT,
    model_version TEXT NOT NULL,
    score REAL NOT NULL,
    score_semantics TEXT NOT NULL,
    threshold REAL NOT NULL,
    threshold_policy TEXT NOT NULL,
    flagged INTEGER NOT NULL CHECK (flagged IN (0,1)),
    policy_version TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    policy_semantics TEXT NOT NULL,
    rule_score REAL NOT NULL CHECK (rule_score >= 0.0 AND rule_score <= 1.0),
    matched_rule_ids TEXT NOT NULL,
    unevaluable_rule_ids TEXT NOT NULL,
    rule_evaluation_complete INTEGER NOT NULL CHECK (rule_evaluation_complete IN (0,1)),
    signal_points INTEGER,
    risk_level TEXT NOT NULL
)
""".strip()

#: 元数据表的规范 DDL。**v1 与 v2 完全相同**——升级只改它唯一那行的 value。
CANONICAL_METADATA_DDL = f"""
CREATE TABLE {_METADATA_TABLE} (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
""".strip()

# --- CW-008：两张幂等表 -------------------------------------------------

#: 幂等请求表的列属性冻结表。
IDEMPOTENCY_REQUEST_COLUMN_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("key_digest", "TEXT", 0, 1),
    ("idempotency_schema_version", "TEXT", 1, 0),
    ("operation", "TEXT", 1, 0),
    ("fingerprint_version", "TEXT", 1, 0),
    ("request_fingerprint", "TEXT", 1, 0),
    ("result_count", "INTEGER", 1, 0),
)

#: 幂等结果关联表的列属性冻结表。复合主键 ``(key_digest, item_index)`` 的 pk 序号为 1、2。
IDEMPOTENCY_RESULT_COLUMN_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("key_digest", "TEXT", 1, 1),
    ("item_index", "INTEGER", 1, 2),
    ("decision_id", "TEXT", 1, 0),
)

IDEMPOTENCY_REQUEST_COLUMNS: tuple[str, ...] = tuple(
    spec[0] for spec in IDEMPOTENCY_REQUEST_COLUMN_SPECS
)
IDEMPOTENCY_RESULT_COLUMNS: tuple[str, ...] = tuple(
    spec[0] for spec in IDEMPOTENCY_RESULT_COLUMN_SPECS
)

#: 幂等请求表的规范 DDL。
#:
#: 摘要长度、operation 域与 ``result_count`` 范围都写成 CHECK，因此"写入一个 63 位摘要"或
#: "operation 写成别的字符串"这类结构性问题在**数据库层**就被拒绝，而不是只靠 Python 侧自觉。
CANONICAL_IDEMPOTENCY_REQUESTS_DDL = f"""
CREATE TABLE {_IDEMPOTENCY_REQUESTS_TABLE} (
    key_digest TEXT PRIMARY KEY,
    idempotency_schema_version TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('{OPERATION_SINGLE}', '{OPERATION_BATCH}')),
    fingerprint_version TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL CHECK (length(request_fingerprint) = 64),
    result_count INTEGER NOT NULL CHECK (result_count >= 1 AND result_count <= {MAX_RESULT_COUNT})
)
""".strip()

#: 幂等结果关联表的规范 DDL：复合主键保证同一 key 内 ``item_index`` 唯一，``decision_id`` 唯一
#: 保证一条决策不会被两个幂等键认领；两个外键都指向真实父行且删除行为为 RESTRICT。
CANONICAL_IDEMPOTENCY_RESULTS_DDL = f"""
CREATE TABLE {_IDEMPOTENCY_RESULTS_TABLE} (
    key_digest TEXT NOT NULL,
    item_index INTEGER NOT NULL CHECK (item_index >= 0 AND item_index <= {MAX_RESULT_COUNT - 1}),
    decision_id TEXT NOT NULL UNIQUE,
    PRIMARY KEY (key_digest, item_index),
    FOREIGN KEY (key_digest) REFERENCES {_IDEMPOTENCY_REQUESTS_TABLE} (key_digest) ON DELETE RESTRICT,
    FOREIGN KEY (decision_id) REFERENCES {_DECISIONS_TABLE} (decision_id) ON DELETE RESTRICT
)
""".strip()


def _create_statement(canonical_ddl: str) -> str:
    """由规范 DDL 派生建表语句（只在表名后插入 ``IF NOT EXISTS``）。"""
    return canonical_ddl.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)


def _normalize_ddl(sql: object) -> str:
    """DDL 的确定性规范化：折叠全部空白为单个空格并去掉首尾。

    只做空白折叠，**不做**任何语义改写（不去注释、不重排、不补全）。因此两份只差空白的
    DDL 视为相同（语义确实相同），而任何字面差异——多一个 CHECK、多一个表级约束或选项、
    列定义有任何改动——都会让精确比较失败。
    """
    return " ".join(str(sql).split())


_CREATE_METADATA_SQL = _create_statement(CANONICAL_METADATA_DDL)
#: 约束写在 schema 里：布尔只能 0/1、rule_score 必须在 [0,1]、decision_id 是主键。
_CREATE_DECISIONS_SQL = _create_statement(CANONICAL_DECISIONS_DDL)
_CREATE_IDEMPOTENCY_REQUESTS_SQL = _create_statement(CANONICAL_IDEMPOTENCY_REQUESTS_DDL)
_CREATE_IDEMPOTENCY_RESULTS_SQL = _create_statement(CANONICAL_IDEMPOTENCY_RESULTS_DDL)


@dataclass(frozen=True)
class SchemaProfile:
    """一个 storage schema 版本的**完整**冻结定义。

    v1 与 v2 共用同一套校验代码，区别只在这份数据里：表集合、每张表的列属性与规范 DDL、
    以及元数据表唯一那行的取值。迁移因此就是"用 v1 档案预校验 → 建新表改版本 → 用 v2 档案复验"，
    不需要任何按版本分叉的校验逻辑。
    """

    schema_version: str
    user_version: int
    tables: frozenset[str]
    column_specs: Mapping[str, tuple[tuple[str, str, int, int], ...]]
    canonical_ddl: Mapping[str, str]
    metadata_rows: tuple[tuple[str, str], ...]


#: 唯一允许迁移的上一版档案。
LEGACY_PROFILE = SchemaProfile(
    schema_version=LEGACY_STORAGE_SCHEMA_VERSION,
    user_version=LEGACY_USER_VERSION,
    tables=frozenset({_DECISIONS_TABLE, _METADATA_TABLE}),
    column_specs={
        _DECISIONS_TABLE: DECISION_COLUMN_SPECS,
        _METADATA_TABLE: METADATA_COLUMN_SPECS,
    },
    canonical_ddl={
        _DECISIONS_TABLE: _normalize_ddl(CANONICAL_DECISIONS_DDL),
        _METADATA_TABLE: _normalize_ddl(CANONICAL_METADATA_DDL),
    },
    metadata_rows=(("schema_version", LEGACY_STORAGE_SCHEMA_VERSION),),
)

#: 当前档案（四张表）。
CURRENT_PROFILE = SchemaProfile(
    schema_version=STORAGE_SCHEMA_VERSION,
    user_version=USER_VERSION,
    tables=frozenset(
        {
            _DECISIONS_TABLE,
            _METADATA_TABLE,
            _IDEMPOTENCY_REQUESTS_TABLE,
            _IDEMPOTENCY_RESULTS_TABLE,
        }
    ),
    column_specs={
        _DECISIONS_TABLE: DECISION_COLUMN_SPECS,
        _METADATA_TABLE: METADATA_COLUMN_SPECS,
        _IDEMPOTENCY_REQUESTS_TABLE: IDEMPOTENCY_REQUEST_COLUMN_SPECS,
        _IDEMPOTENCY_RESULTS_TABLE: IDEMPOTENCY_RESULT_COLUMN_SPECS,
    },
    canonical_ddl={
        _DECISIONS_TABLE: _normalize_ddl(CANONICAL_DECISIONS_DDL),
        _METADATA_TABLE: _normalize_ddl(CANONICAL_METADATA_DDL),
        _IDEMPOTENCY_REQUESTS_TABLE: _normalize_ddl(CANONICAL_IDEMPOTENCY_REQUESTS_DDL),
        _IDEMPOTENCY_RESULTS_TABLE: _normalize_ddl(CANONICAL_IDEMPOTENCY_RESULTS_DDL),
    },
    metadata_rows=(("schema_version", STORAGE_SCHEMA_VERSION),),
)

#: 表名 → 规范化后的规范 DDL（当前版本）。保留旧名字供既有测试与文档引用。
_CANONICAL_DDL_BY_TABLE: dict[str, str] = dict(CURRENT_PROFILE.canonical_ddl)

_INSERT_SQL = (
    f"INSERT INTO {_DECISIONS_TABLE} ({', '.join(DECISION_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(DECISION_COLUMNS))})"
)
_SELECT_SQL = (
    f"SELECT {', '.join(DECISION_COLUMNS)} FROM {_DECISIONS_TABLE} WHERE decision_id = ?"
)

_SELECT_IDEMPOTENCY_REQUEST_SQL = (
    f"SELECT {', '.join(IDEMPOTENCY_REQUEST_COLUMNS)} "
    f"FROM {_IDEMPOTENCY_REQUESTS_TABLE} WHERE key_digest = ?"
)
_SELECT_IDEMPOTENCY_RESULTS_SQL = (
    f"SELECT item_index, decision_id FROM {_IDEMPOTENCY_RESULTS_TABLE} "
    f"WHERE key_digest = ? ORDER BY item_index"
)
_INSERT_IDEMPOTENCY_REQUEST_SQL = (
    f"INSERT INTO {_IDEMPOTENCY_REQUESTS_TABLE} ({', '.join(IDEMPOTENCY_REQUEST_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(IDEMPOTENCY_REQUEST_COLUMNS))})"
)
_INSERT_IDEMPOTENCY_RESULT_SQL = (
    f"INSERT INTO {_IDEMPOTENCY_RESULTS_TABLE} (key_digest, item_index, decision_id) "
    f"VALUES (?, ?, ?)"
)


class StorageUnavailableError(StorageLoadError):
    """运行期读写失败（磁盘、锁、约束冲突等）。

    对外折叠为 503 ``storage_unavailable``：调用方可以为"存储暂时不可用"重试，
    但**不会**拿到任何未持久化的结果。
    """

    def __init__(self, detail: str) -> None:
        super().__init__(StorageErrorCode.UNAVAILABLE, detail)


#: 幂等查找/提交的结果标记。``lookup`` 用前三个，``commit`` 用 ``created`` / ``replayed`` / ``conflict``。
IDEMPOTENCY_MISS = "miss"
IDEMPOTENCY_REPLAY = "replay"
IDEMPOTENCY_CONFLICT = "conflict"
IDEMPOTENCY_CREATED = "created"
IDEMPOTENCY_REPLAYED = "replayed"


@dataclass(frozen=True)
class IdempotencyHit:
    """一次幂等查找或提交的结果。

    ``records`` 是**已提交的存储快照**（重放时来自数据库，不是本次的候选对象），
    顺序与首次提交的 ``item_index`` 一致。冲突与未命中时为空。
    """

    status: str
    records: tuple[StoredDecision, ...] = ()


@runtime_checkable
class DecisionRepository(Protocol):
    """存储仓储接口：真实 SQLite 实现与测试 fake 都实现它。"""

    def insert_many(self, records: Sequence[StoredDecision]) -> None:
        """在一个事务里写入全部记录；全部成功或全部回滚。"""
        ...

    def get(self, decision_id: str) -> StoredDecision | None:
        """按 decision ID 读回记录；不存在返回 ``None``。"""
        ...

    def lookup_idempotent(
        self, key_digest: str, operation: str, request_fingerprint: str
    ) -> IdempotencyHit:
        """按 key 摘要查已有映射：``miss`` / ``replay`` / ``conflict``。"""
        ...

    def commit_idempotent(
        self,
        key_digest: str,
        operation: str,
        request_fingerprint: str,
        records: Sequence[StoredDecision],
    ) -> IdempotencyHit:
        """在一个事务里解决竞态并写入：``created`` / ``replayed`` / ``conflict``。"""
        ...

    def close(self) -> None:
        """释放资源（本实现下是幂等的空操作，因为连接是逐操作短生命周期的）。"""
        ...


class SqliteDecisionRepository:
    """标准库 ``sqlite3`` 实现。"""

    def __init__(self, path: Path | str, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
        self._path = Path(path)
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    # --- 连接 -----------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """打开一个短生命周期连接，应用冻结 PRAGMA，并在退出时**必定关闭**。"""
        if self._closed:
            raise StorageUnavailableError("存储已关闭")
        try:
            connection = sqlite3.connect(self._path, isolation_level=None)
        except sqlite3.Error as error:
            raise StorageUnavailableError(f"无法打开数据库：{type(error).__name__}") from error
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA foreign_keys = {FOREIGN_KEYS}")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute(f"PRAGMA journal_mode = {JOURNAL_MODE}")
            connection.execute(f"PRAGMA synchronous = {SYNCHRONOUS}")
            yield connection
        finally:
            connection.close()

    # --- 初始化 ---------------------------------------------------------

    def initialize(self) -> None:
        """按冻结契约准备数据库：新建 v2、验证 v2，或把严格 v1 原子迁移到 v2。

        三种启动状态，别的都不接受：

        1. **新文件** → 直接创建严格 v2；
        2. **已有严格 v2** → 只验证，**不写入**（重复启动零写入）；
        3. **已有完整合法 v1** → 执行唯一允许的那次 v1→v2 迁移。

        其它 ``user_version``、结构漂移或损坏文件一律失败关闭，不猜测、不覆盖、不删除、不跳迁。

        Raises:
            StorageLoadError: 目录不可创建、数据库不可打开、结构未知/损坏或版本不符。
                **失败时不得改写、删除或迁移已有文件**（迁移中途失败只回滚自己那个事务）。
        """
        existed = self._path.is_file() and self._path.stat().st_size > 0
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise StorageLoadError(
                StorageErrorCode.UNAVAILABLE, f"无法创建存储目录：{type(error).__name__}"
            ) from error

        try:
            with self._connect() as connection:
                if not existed:
                    self._create_schema(connection)
                    return
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version == USER_VERSION:
                    self._validate_profile(connection, CURRENT_PROFILE)
                elif version == LEGACY_USER_VERSION:
                    self._migrate_legacy_to_current(connection)
                else:
                    raise StorageLoadError(
                        StorageErrorCode.SCHEMA_MISMATCH, "user_version 不是任何冻结版本"
                    )
        except sqlite3.Error as error:
            # 非 SQLite 文件、被截断的文件、只读介质等都折叠成"损坏/不可用"。
            raise StorageLoadError(
                StorageErrorCode.CORRUPT, f"数据库不可用：{type(error).__name__}"
            ) from error

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        """在**一个事务**里建出当前版本的全部四张表，并写入版本标记。"""
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(_CREATE_METADATA_SQL)
            connection.execute(_CREATE_DECISIONS_SQL)
            connection.execute(_CREATE_IDEMPOTENCY_REQUESTS_SQL)
            connection.execute(_CREATE_IDEMPOTENCY_RESULTS_SQL)
            connection.execute(
                f"INSERT OR IGNORE INTO {_METADATA_TABLE} (key, value) VALUES (?, ?)",
                ("schema_version", STORAGE_SCHEMA_VERSION),
            )
            # USER_VERSION 是冻结的模块常量（不是调用方数据），因此可以安全拼接。
            connection.execute(f"PRAGMA user_version = {USER_VERSION}")
            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise

    # --- v1 -> v2：唯一允许的迁移 ---------------------------------------

    def _migrate_legacy_to_current(self, connection: sqlite3.Connection) -> None:
        """把**完整合法**的 v1 库在一个 ``BEGIN IMMEDIATE`` 中升级为 v2。

        步骤是冻结的：先按 v1 档案做完整预校验（任何漂移都在**动任何东西之前**失败），
        再在同一个事务里建两张幂等表、改写元数据版本、提升 ``user_version``，最后 commit，
        并**在 commit 之后**按完整 v2 契约复验一次——因为"写完了"不等于"写对了"。

        事务中途失败只回滚本次事务：``decisions`` 的既有行、v1 表结构与文件其余部分都不受影响；
        绝不出现"半迁移"（只有一张新表、或版本已改而表未建）。
        """
        # 1) 预校验：不通过就原样失败关闭，之后的任何写入都不会发生。
        #    结构合法还不够——既有记录必须**读得出来**，否则不得带着坏行进 v2。
        self._validate_profile(connection, LEGACY_PROFILE)
        self._validate_legacy_records(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(_CREATE_IDEMPOTENCY_REQUESTS_SQL)
            connection.execute(_CREATE_IDEMPOTENCY_RESULTS_SQL)
            connection.execute(
                f"UPDATE {_METADATA_TABLE} SET value = ? WHERE key = ?",
                (STORAGE_SCHEMA_VERSION, "schema_version"),
            )
            connection.execute(f"PRAGMA user_version = {USER_VERSION}")
            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise
        # 2) 复验：commit 之后的库必须逐项满足完整 v2 契约。
        self._validate_profile(connection, CURRENT_PROFILE)

    def _validate_existing(self, connection: sqlite3.Connection) -> None:
        """按**当前**版本的冻结契约验证已有数据库（保留的兼容入口）。"""
        self._validate_profile(connection, CURRENT_PROFILE)

    def _validate_legacy_records(self, connection: sqlite3.Connection) -> None:
        """迁移前逐行确认既有决策记录仍然可读、且满足记录契约。

        结构完全相等并不保证**行内容**完好：一个被手工改坏的 ``risk_level`` 或非 0/1 布尔
        在 schema 层完全看不出来。迁移是要把库交给下一版使用的，因此必须在此之前确认
        "迁过去的东西是能用的"，而不是把损坏一并带入 v2。
        """
        for row in connection.execute(f"SELECT {', '.join(DECISION_COLUMNS)} FROM {_DECISIONS_TABLE}"):
            try:
                StoredDecision.from_row(dict(row))
            except StorageContractError as error:
                raise StorageLoadError(
                    StorageErrorCode.CONTRACT_VIOLATION, f"既有决策记录已损坏：{error}"
                ) from error

    def _validate_profile(
        self, connection: sqlite3.Connection, profile: SchemaProfile
    ) -> None:
        """已有数据库必须与给定版本的冻结定义**逐项精确匹配**；任一不符即失败关闭。

        校验的是"结构是否**完全等于**冻结定义"，而不是"必需的东西在不在"：多一张表、
        多一个视图/触发器/手工索引、元数据多一行、列的顺序/类型/NOT NULL/主键被改、
        多一个约束，都属于结构漂移，必须失败关闭——**不删除、不覆盖、不迁移**。
        """
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != profile.user_version:
            raise StorageLoadError(
                StorageErrorCode.SCHEMA_MISMATCH, "user_version 不是冻结值"
            )
        self._validate_schema_objects(connection, profile)
        for table in sorted(profile.tables):
            self._validate_table_columns(
                connection, table, profile.column_specs[table], f"表 {table}"
            )
            self._validate_table_ddl(connection, table, profile)
        self._validate_metadata_rows(connection, profile)

    def _validate_schema_objects(
        self, connection: sqlite3.Connection, profile: SchemaProfile
    ) -> None:
        """用户表集合必须**精确相等**；不得有视图、触发器或手工索引。"""
        objects = [
            (str(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT type, name, tbl_name FROM sqlite_master WHERE name NOT LIKE ?",
                (f"{_AUTO_INDEX_PREFIX}%",),
            )
        ]
        tables = {name for kind, name, _ in objects if kind == "table"}
        if tables != set(profile.tables):
            raise StorageLoadError(
                StorageErrorCode.SCHEMA_MISMATCH, "用户表集合与冻结 schema 不一致"
            )
        for kind, name, table in objects:
            if kind == "table":
                continue
            if kind == "index" and name.startswith(_AUTO_INDEX_PREFIX):
                continue
            # view / trigger / 手工 index 都是结构漂移（内部自动索引已在上面的过滤中放行）。
            raise StorageLoadError(
                StorageErrorCode.SCHEMA_MISMATCH,
                f"schema 中存在未声明的 {kind} 对象（归属表 {table}）",
            )

    def _validate_table_columns(
        self,
        connection: sqlite3.Connection,
        table: str,
        specs: tuple[tuple[str, str, int, int], ...],
        label: str,
    ) -> None:
        """列必须**按顺序**逐项匹配名字、类型、NOT NULL 与主键属性。"""
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != specs:
            raise StorageLoadError(
                StorageErrorCode.SCHEMA_MISMATCH, f"{label}的列定义与冻结 schema 不一致"
            )

    def _validate_metadata_rows(
        self, connection: sqlite3.Connection, profile: SchemaProfile
    ) -> None:
        """元数据**只能**含冻结的那一条 ``schema_version`` 记录（取值随版本不同）。"""
        rows = tuple(
            (str(row[0]), str(row[1]))
            for row in connection.execute(f"SELECT key, value FROM {_METADATA_TABLE}")
        )
        if rows != profile.metadata_rows:
            raise StorageLoadError(
                StorageErrorCode.SCHEMA_MISMATCH, "元数据记录与冻结定义不一致"
            )

    def _validate_table_ddl(
        self, connection: sqlite3.Connection, table: str, profile: SchemaProfile
    ) -> None:
        """表的**完整 DDL** 必须与冻结规范精确相等——不是"包含必需片段"。

        约束（CHECK、PRIMARY KEY、UNIQUE、FOREIGN KEY）与表级选项都不是列属性，``PRAGMA
        table_info`` 看不到它们，因此只能回到 ``sqlite_master.sql`` 做**整串比对**。只折叠空白，
        不做任何语义改写，所以：

        - 拿掉任何一个原有约束 → 字面不同 → 失败关闭；
        - **增加**一个额外 CHECK / UNIQUE / 表级选项 → 字面不同 → 同样失败关闭。

        后者正是"包含式校验"漏掉的一类：保留全部必要约束再加一个额外约束时，旧实现会放行。
        """
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        stored = _normalize_ddl(row[0]) if row is not None else ""
        if stored != profile.canonical_ddl[table]:
            raise StorageLoadError(
                StorageErrorCode.SCHEMA_MISMATCH, "表的 DDL 与冻结规范不一致"
            )

    # --- 写入 -----------------------------------------------------------

    def insert_many(self, records: Sequence[StoredDecision]) -> None:
        """在**一个事务**里写入全部记录。空序列是空操作。

        Raises:
            StorageUnavailableError: 批次内 ID 重复，或任一条 insert/commit 失败。
                此时事务已全部回滚，**没有任何**记录落库。
        """
        payload = list(records)
        if not payload:
            return
        identifiers = [record.decision_id for record in payload]
        if len(set(identifiers)) != len(identifiers):
            raise StorageUnavailableError("批次内 decision_id 重复")

        rows = [
            tuple(record.to_row()[column] for column in DECISION_COLUMNS) for record in payload
        ]
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.executemany(_INSERT_SQL, rows)
                    connection.commit()
                except sqlite3.Error as error:
                    connection.rollback()
                    raise StorageUnavailableError(
                        f"写入失败并已回滚：{type(error).__name__}"
                    ) from error
        except StorageUnavailableError:
            raise
        except sqlite3.Error as error:
            raise StorageUnavailableError(f"写入失败：{type(error).__name__}") from error

    # --- 读取 -----------------------------------------------------------

    def get(self, decision_id: str) -> StoredDecision | None:
        """按 ID 读回并**重新校验**记录。

        Raises:
            StorageUnavailableError: 数据库不可读。
            StorageLoadError: 行结构损坏（``storage_contract_violation``）。
                错误消息只说明违反了什么规则，不回显行内容。
        """
        try:
            with self._connect() as connection:
                row = connection.execute(_SELECT_SQL, (decision_id,)).fetchone()
        except sqlite3.Error as error:
            raise StorageUnavailableError(f"读取失败：{type(error).__name__}") from error
        if row is None:
            return None
        try:
            return StoredDecision.from_row(dict(row))
        except StorageContractError as error:
            raise StorageLoadError(
                StorageErrorCode.CONTRACT_VIOLATION, f"存储的记录结构损坏：{error}"
            ) from error

    # --- 幂等：查找与原子提交 -------------------------------------------

    def lookup_idempotent(
        self, key_digest: str, operation: str, request_fingerprint: str
    ) -> IdempotencyHit:
        """只读查找已有幂等映射；不写任何东西。"""
        try:
            with self._connect() as connection:
                return _read_idempotent(connection, key_digest, operation, request_fingerprint)
        except sqlite3.Error as error:
            raise StorageUnavailableError(f"幂等查找失败：{type(error).__name__}") from error

    def commit_idempotent(
        self,
        key_digest: str,
        operation: str,
        request_fingerprint: str,
        records: Sequence[StoredDecision],
    ) -> IdempotencyHit:
        """在**一个** ``BEGIN IMMEDIATE`` 事务里解决并发竞态并写入首次结果。

        这是"可能重复评分、但只提交一组结果"的落点：先查再写，因此两个并发请求即使都已经
        完成评分，也只有一个能把候选记录写进去；另一个在事务内看到已提交映射后改为安全的重放
        或冲突，**不会**留下自己的候选记录。

        Returns:
            ``created``（本次写入，``records`` 为本次候选）、
            ``replayed``（同键同请求，``records`` 为**库里已提交的**快照）、
            ``conflict``（同键异请求）。

        Raises:
            StorageUnavailableError: 任一步 insert/commit 失败（此时已全部回滚）。
            IdempotencyContractError: 已有映射损坏。
        """
        payload = list(records)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    hit = _read_idempotent(
                        connection, key_digest, operation, request_fingerprint
                    )
                    if hit.status == IDEMPOTENCY_CONFLICT:
                        connection.rollback()
                        return hit
                    if hit.status == IDEMPOTENCY_REPLAY:
                        # 竞态输家：不写自己的候选，直接返回赢家已提交的快照。
                        connection.rollback()
                        return IdempotencyHit(IDEMPOTENCY_REPLAYED, hit.records)
                    _insert_decisions(connection, payload)
                    connection.execute(
                        _INSERT_IDEMPOTENCY_REQUEST_SQL,
                        (
                            key_digest,
                            IDEMPOTENCY_SCHEMA_VERSION,
                            operation,
                            FINGERPRINT_VERSION,
                            request_fingerprint,
                            len(payload),
                        ),
                    )
                    connection.executemany(
                        _INSERT_IDEMPOTENCY_RESULT_SQL,
                        [
                            (key_digest, index, record.decision_id)
                            for index, record in enumerate(payload)
                        ],
                    )
                    connection.commit()
                except (sqlite3.Error, IdempotencyContractError):
                    connection.rollback()
                    raise
                except StorageUnavailableError:
                    connection.rollback()
                    raise
        except sqlite3.Error as error:
            raise StorageUnavailableError(
                f"幂等提交失败并已回滚：{type(error).__name__}"
            ) from error
        return IdempotencyHit(IDEMPOTENCY_CREATED, tuple(payload))

    def close(self) -> None:
        """标记关闭；连接是逐操作短生命周期的，因此这里是幂等的空操作。"""
        self._closed = True


def _insert_decisions(
    connection: sqlite3.Connection, records: Sequence[StoredDecision]
) -> None:
    """把全部记录写进当前事务（调用方负责事务边界）。"""
    if not records:
        return
    connection.executemany(
        _INSERT_SQL,
        [tuple(record.to_row()[column] for column in DECISION_COLUMNS) for record in records],
    )


def _read_idempotent(
    connection: sqlite3.Connection, key_digest: str, operation: str, request_fingerprint: str
) -> IdempotencyHit:
    """读取并**逐项重新校验**已有幂等映射。

    一律以数据库内容为准：摘要形状、版本常量、operation 域、``result_count`` 与
    ``item_index`` 是否连续、``decision_id`` 是否规范 UUID4、外键指向的决策是否真的存在且
    自身契约完好。任何一项不符都抛 :class:`IdempotencyContractError`，**不修理、不猜测**。

    **完整校验先于 replay/conflict 判定**：只要这个 key 的映射存在，就先把结果行与关联决策
    全部读出来验一遍，再比较 operation 与 fingerprint。否则一个损坏的映射会因为"这次请求恰好
    是异请求"而被报成 409，把服务端的存储损坏伪装成调用方的冲突——那是错误分类，不是宽容。

    Raises:
        IdempotencyContractError: 映射或关联损坏（消息不含任何摘要、ID 或行内容）。
    """
    row = connection.execute(_SELECT_IDEMPOTENCY_REQUEST_SQL, (key_digest,)).fetchone()
    if row is None:
        return IdempotencyHit(IDEMPOTENCY_MISS)
    if not is_digest(key_digest) or not is_digest(str(row["request_fingerprint"])):
        raise IdempotencyContractError("幂等映射中的摘要形状非法")
    if str(row["idempotency_schema_version"]) != IDEMPOTENCY_SCHEMA_VERSION:
        raise IdempotencyContractError("幂等映射的 schema 版本不是冻结值")
    if str(row["fingerprint_version"]) != FINGERPRINT_VERSION:
        raise IdempotencyContractError("幂等映射的指纹版本不是冻结值")
    stored_operation = str(row["operation"])
    if stored_operation not in OPERATIONS:
        raise IdempotencyContractError("幂等映射的 operation 不是冻结值")
    result_count = row["result_count"]
    if isinstance(result_count, bool) or not isinstance(result_count, int):
        raise IdempotencyContractError("幂等映射的 result_count 不是整数")
    if not 1 <= result_count <= MAX_RESULT_COUNT:
        raise IdempotencyContractError("幂等映射的 result_count 越界")

    # 先完整校验结果行与关联决策（损坏必须在**任何**分支上都是 500）。
    records = _read_results(connection, key_digest, result_count)
    if stored_operation != operation or str(row["request_fingerprint"]) != request_fingerprint:
        return IdempotencyHit(IDEMPOTENCY_CONFLICT)
    return IdempotencyHit(IDEMPOTENCY_REPLAY, records)


def _read_results(
    connection: sqlite3.Connection, key_digest: str, result_count: int
) -> tuple[StoredDecision, ...]:
    """按 ``item_index`` 读回有序决策快照，并校验关联完整性。"""
    rows = [
        (row["item_index"], str(row["decision_id"]))
        for row in connection.execute(_SELECT_IDEMPOTENCY_RESULTS_SQL, (key_digest,))
    ]
    if len(rows) != result_count:
        raise IdempotencyContractError("幂等结果行数与 result_count 不一致")
    indices = [index for index, _ in rows]
    if indices != list(range(result_count)):
        raise IdempotencyContractError("幂等结果的 item_index 不连续")
    records: list[StoredDecision] = []
    for _, decision_id in rows:
        record = _read_decision(connection, decision_id)
        records.append(record)
    return tuple(records)


def _read_decision(connection: sqlite3.Connection, decision_id: str) -> StoredDecision:
    """读一条决策；关联缺失或记录损坏都算**幂等契约**违规。

    ``StoredDecision.from_row()`` 抛的是记录层的 :class:`StorageContractError`；这里必须把它
    转成 :class:`IdempotencyContractError`，否则它会一路冒到路由层被折叠成通用 ``internal_error``，
    而冻结契约要求这类损坏稳定映射为 ``500 idempotency_contract_violation``——
    错误分类是契约的一部分，不是实现细节。
    """
    if not is_canonical_decision_id(decision_id):
        raise IdempotencyContractError("幂等结果引用了非规范 UUID4")
    row = connection.execute(_SELECT_SQL, (decision_id,)).fetchone()
    if row is None:
        raise IdempotencyContractError("幂等结果引用了不存在的决策记录")
    try:
        record = StoredDecision.from_row(dict(row))
    except StorageContractError as error:
        raise IdempotencyContractError(f"幂等结果关联的决策记录已损坏：{error}") from error
    if record.decision_id != decision_id:  # pragma: no cover - 主键查询下不可能
        raise IdempotencyContractError("幂等结果与决策 ID 不匹配")
    return record


def open_repository(path: Path | str, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> SqliteDecisionRepository:
    """构造并初始化一个 repository。

    **位置策略不在这里**：调用方必须先通过
    :func:`ai_detector.storage.config.resolve_storage_path` 把配置里的项目相对路径解析成
    安全路径。生产入口是 ``ai_detector.service.runtime.load_storage_repository``；
    测试可以直接传临时路径来单独验证 repository 行为。
    """
    repository = SqliteDecisionRepository(path, busy_timeout_ms=busy_timeout_ms)
    repository.initialize()
    return repository


def row_to_dict(row: Any) -> dict[str, Any]:
    """把 ``sqlite3.Row`` 转成普通字典（供需要逐字段处理的调用方使用）。"""
    return {key: row[key] for key in row.keys()}
