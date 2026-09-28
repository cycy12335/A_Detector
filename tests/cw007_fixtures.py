"""CW-007 测试夹具：存储配置构造、fake repository、可注入时钟与 ID factory。

设计要点：

- **生产路径入口不放宽**：应用级测试也把数据库声明在**项目相对**的
  ``artifacts/runtime/*.sqlite3`` 下（该目录已被 Git 忽略），因此走的是与生产完全相同的
  路径校验链路；只有直接测试 repository 本身时才使用任意临时路径。
- **fake repository** 只实现 ``insert_many`` / ``get`` / ``close``，因此可以在不碰 SQLite 的
  情况下精确断言"写入了几次""是否回滚""GET 有没有触发模型或规则"。
- **注入式时钟与 ID factory**：时间与 UUID 由测试给定，记录因此是逐字段可断言的常量。
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

if TYPE_CHECKING:  # pragma: no cover - 仅供类型标注
    import sqlite3

from ai_detector.config import PROJECT_ROOT
from ai_detector.storage.config import STORAGE_SCHEMA_VERSION
from ai_detector.rules.engine import RiskAssessment, RuleEvaluationResult

#: 生产允许的数据库目录（项目相对）。
STORAGE_DIR = "artifacts/runtime"

#: 测试数据库文件名前缀：会话结束时会按此前缀清理，避免在仓库里留下临时数据库。
TEST_DB_PREFIX = "cw007_test_"

#: 测试用决策 ID 基线（规范小写 UUID4）。
DECISION_ID_A = "11111111-1111-4111-8111-111111111111"
DECISION_ID_B = "22222222-2222-4222-8222-222222222222"
DECISION_ID_C = "33333333-3333-4333-8333-333333333333"
DECISION_ID_D = "44444444-4444-4444-8444-444444444444"

#: 测试用固定时间（UTC RFC 3339，后缀 Z）。
CREATED_AT = "2026-09-25T04:05:06Z"
CREATED_AT_LATER = "2026-09-25T04:05:07Z"


def storage_section(
    relative_path: str, *, schema_version: str = STORAGE_SCHEMA_VERSION
) -> dict:
    """构造 ``configs/service.yaml`` 的 storage 段。

    默认取**当前**冻结版本（CW-008 起是 ``cw008_storage_v2``）：夹具必须跟着冻结常量走，
    否则每次升版都要在测试里散落地改字面量，迟早漏掉一处。
    """
    return {"path": relative_path, "schema_version": schema_version}


def _test_db_stem(tmp_path: Path) -> str:
    """由 ``tmp_path`` 的真实路径派生一个**唯一且稳定**的文件名主干。

    刻意不用 ``tmp_path.name``：pytest 会把目录名截断到 30 字符，于是名字相近的测试会
    撞到同一个数据库文件，出现"上一个测试留下的记录被下一个测试读到"这种幽灵失败。
    对完整路径取短摘要既唯一又在该测试内稳定（同一测试内多次调用得到同一路径）。
    """
    digest = hashlib.sha1(str(tmp_path).encode("utf-8")).hexdigest()[:12]
    return f"{TEST_DB_PREFIX}{digest}"


def staged_db_relative(tmp_path: Path, name: str = "decisions.sqlite3") -> str:
    """返回项目相对的测试数据库路径（位于 ``artifacts/runtime/`` 下，被 Git 忽略）。"""
    return f"{STORAGE_DIR}/{_test_db_stem(tmp_path)}_{name}"


def staged_db_path(tmp_path: Path, name: str = "decisions.sqlite3") -> Path:
    """返回与 :func:`staged_db_relative` 对应的绝对路径。"""
    return PROJECT_ROOT / staged_db_relative(tmp_path, name)


@contextmanager
def sqlite_connection(path: Path) -> Iterator["sqlite3.Connection"]:
    """打开一个**必定关闭**的连接。

    刻意不用 ``with sqlite3.connect(...)``：那个上下文管理器只提交/回滚，**不关闭**连接，
    于是在 Windows 上文件会一直被占用，测试收尾删不掉。这里保留它的提交语义，
    并**额外**保证连接关闭。
    """
    import sqlite3

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def remove_database(path: Path) -> None:
    """尽力删除数据库及其 sidecar（只删除调用方明确给出的路径）。

    清理失败**不得**让测试失败：文件残留在 tmp 目录里无害，而占用导致的 PermissionError
    会把一个真实的断言失败掩盖成收尾错误。
    """
    for suffix in ("", "-journal", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        try:
            if candidate.is_file():
                candidate.unlink()
        except OSError:
            pass


class FakeDecisionRepository:
    """实现 repository 协议的测试替身：内存字典 + 可注入故障 + 调用计数。"""

    def __init__(
        self,
        *,
        insert_error: Exception | None = None,
        get_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.records: dict[str, Any] = {}
        self.insert_calls: list[list[Any]] = []
        self.get_calls: list[str] = []
        self.close_calls = 0
        self._insert_error = insert_error
        self._get_error = get_error
        self._close_error = close_error

    @property
    def insert_count(self) -> int:
        return len(self.insert_calls)

    def insert_many(self, records: Sequence[Any]) -> None:
        payload = list(records)
        self.insert_calls.append(payload)
        if self._insert_error is not None:
            # 真实实现要么整批写入要么全回滚，因此这里也不留下任何记录。
            raise self._insert_error
        for record in payload:
            self.records[record.decision_id] = record

    def get(self, decision_id: str) -> Any:
        self.get_calls.append(decision_id)
        if self._get_error is not None:
            raise self._get_error
        return self.records.get(decision_id)

    def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:  # pragma: no cover - 仅用于覆盖关闭异常路径
            raise self._close_error


class RecordingStorageLoader:
    """记录调用次数的 storage 加载器；可选地抛出预设异常。"""

    def __init__(self, repository: Any = None, error: Exception | None = None) -> None:
        self.repository = repository if repository is not None else FakeDecisionRepository()
        self.error = error
        self.calls = 0

    def __call__(self, config: Any) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.repository


class SequentialIdFactory:
    """确定性 decision ID 工厂（按顺序取给定 ID，用尽后回落到 uuid4）。"""

    def __init__(self, ids: Sequence[str]) -> None:
        self._ids = list(ids)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self._ids:
            return self._ids.pop(0)
        import uuid

        return str(uuid.uuid4())


class FixedClock:
    """确定性 UTC 时钟：每次调用返回给定序列中的下一个时间戳（用尽后重复最后一个）。"""

    def __init__(self, timestamps: Sequence[str]) -> None:
        self._timestamps = list(timestamps)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if not self._timestamps:
            return CREATED_AT
        if len(self._timestamps) == 1:
            return self._timestamps[0]
        return self._timestamps.pop(0)


# --- 记录构造：直接由真实融合函数产出，避免手写字段与产品逻辑漂移 ---


def build_assessment(*, flagged: bool, matched: int, complete: bool = True) -> RiskAssessment:
    """用**真实** ``fuse()`` 构造一个评估结果，字段不可能与冻结真值表不一致。"""
    from ai_detector.rules.config import FROZEN_RULES
    from ai_detector.rules.engine import RuleEvaluation, RuleEvaluationResult, fuse

    evaluations = []
    for index, rule in enumerate(FROZEN_RULES):
        if index < matched:
            evaluations.append(RuleEvaluation(rule.rule_id, matched=True, evaluable=True))
        else:
            evaluations.append(
                RuleEvaluation(rule.rule_id, matched=False, evaluable=complete)
            )
    return fuse(RuleEvaluationResult(evaluations=tuple(evaluations)), flagged=flagged)


def evaluation_result(*, matched: int, complete: bool = True) -> RuleEvaluationResult:
    """构造一个规则求值结果（供 fake engine 使用）。"""
    from ai_detector.rules.config import FROZEN_RULES
    from ai_detector.rules.engine import RuleEvaluation, RuleEvaluationResult

    evaluations = []
    for index, rule in enumerate(FROZEN_RULES):
        if index < matched:
            evaluations.append(RuleEvaluation(rule.rule_id, matched=True, evaluable=True))
        else:
            evaluations.append(RuleEvaluation(rule.rule_id, matched=False, evaluable=complete))
    return RuleEvaluationResult(evaluations=tuple(evaluations))
