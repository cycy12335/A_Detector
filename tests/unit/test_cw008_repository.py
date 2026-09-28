"""CW-008 repository 幂等接口的单元测试：查找、原子提交、竞态与损坏映射。

要点：

- 未命中只读不写；``created`` 一次事务写入 decisions + 两张映射表；
- 同键同请求再提交必须**不新增任何行**，返回库里**已提交**的快照；
- 同键异请求/异 operation 必须冲突且零新增；
- 任一步失败全部回滚（decisions 与映射都不留）；
- 损坏映射必须抛 ``IdempotencyContractError``，不修理、不猜测。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_detector.storage.idempotency import (
    FINGERPRINT_VERSION,
    IDEMPOTENCY_SCHEMA_VERSION,
    OPERATION_BATCH,
    OPERATION_SINGLE,
    IdempotencyContractError,
    key_digest,
    request_fingerprint,
    single_payload,
)
from ai_detector.storage.sqlite import (
    IDEMPOTENCY_CONFLICT,
    IDEMPOTENCY_CREATED,
    IDEMPOTENCY_MISS,
    IDEMPOTENCY_REPLAYED,
    open_repository,
)
from tests.cw007_fixtures import remove_database
from tests.cw008_fixtures import (
    KEY_ONE,
    KEY_THREE,
    KEY_TWO,
    build_record,
    read_all,
    single_key_and_fingerprint,
    tamper,
    user_version,
)

V1_IDS = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
)

FEATURES = {"TransactionAmt": 100.0, "C13": 1.0}
OTHER_FEATURES = {"TransactionAmt": 200.0, "C13": 1.0}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "cw008_repo.sqlite3"
    yield path
    remove_database(path)


@pytest.fixture
def repository(db_path: Path):
    repo = open_repository(db_path)
    yield repo
    repo.close()


def _records(count: int, *, offset: int = 0) -> list:
    return [
        build_record(V1_IDS[index], client_item_id=f"item-{index + offset}")
        for index in range(count)
    ]


def _rows(db_path: Path, table: str, columns: tuple[str, ...], order: str = "") -> list[tuple]:
    return read_all(db_path, table, columns, order)


# --- 未命中：只读 -------------------------------------------------------


def test_lookup_on_an_empty_database_is_a_miss(repository, db_path: Path) -> None:
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    assert repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint).status == IDEMPOTENCY_MISS
    assert _rows(db_path, "idempotency_requests", ("key_digest",)) == []
    assert _rows(db_path, "decisions", ("decision_id",)) == []


# --- 首次提交 -----------------------------------------------------------


def test_commit_creates_decisions_and_both_mapping_tables(repository, db_path: Path) -> None:
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    records = _records(1)
    hit = repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, records)
    assert hit.status == IDEMPOTENCY_CREATED
    assert [record.decision_id for record in hit.records] == [V1_IDS[0]]

    assert len(_rows(db_path, "decisions", ("decision_id",))) == 1
    request_rows = _rows(
        db_path,
        "idempotency_requests",
        ("key_digest", "idempotency_schema_version", "operation", "fingerprint_version",
         "request_fingerprint", "result_count"),
    )
    assert request_rows == [
        (digest, IDEMPOTENCY_SCHEMA_VERSION, OPERATION_SINGLE, FINGERPRINT_VERSION, fingerprint, 1)
    ]
    result_rows = _rows(db_path, "idempotency_results", ("key_digest", "item_index", "decision_id"), "item_index")
    assert result_rows == [(digest, 0, V1_IDS[0])]


def test_commit_batch_keeps_order_in_the_mapping(repository, db_path: Path) -> None:
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    payload = {"items": [{"client_item_id": f"item-{i}", "features": FEATURES} for i in range(3)]}
    digest = key_digest(KEY_ONE)
    fingerprint = request_fingerprint(OPERATION_BATCH, payload)
    hit = repository.commit_idempotent(digest, OPERATION_BATCH, fingerprint, _records(3))
    assert hit.status == IDEMPOTENCY_CREATED
    results = _rows(db_path, "idempotency_results", ("item_index", "decision_id"), "item_index")
    assert results == [(0, V1_IDS[0]), (1, V1_IDS[1]), (2, V1_IDS[2])]


def test_plain_insert_many_does_not_touch_the_mapping_tables(repository, db_path: Path) -> None:
    repository.insert_many(_records(2))
    assert len(_rows(db_path, "decisions", ("decision_id",))) == 2
    assert _rows(db_path, "idempotency_requests", ("key_digest",)) == []
    assert _rows(db_path, "idempotency_results", ("key_digest",)) == []


# --- 重放 ---------------------------------------------------------------


def test_replay_returns_the_committed_snapshot_from_storage(repository, db_path: Path) -> None:
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    created = repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(1))
    hit = repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint)
    assert hit.status == "replay"
    assert hit.records == created.records
    assert hit.records[0] == repository.get(V1_IDS[0]), "重放必须给库里那一份，而不是候选对象"


def test_second_commit_of_the_same_request_replays_without_writing(repository, db_path: Path) -> None:
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(1))
    before = _rows(db_path, "decisions", ("decision_id",))

    # 候选记录故意换成别的 ID：必须被忽略，返回库里已提交的那一份。
    hit = repository.commit_idempotent(
        digest, OPERATION_SINGLE, fingerprint, _records(1, offset=10)
    )
    assert hit.status == IDEMPOTENCY_REPLAYED
    assert [record.decision_id for record in hit.records] == [V1_IDS[0]]
    assert _rows(db_path, "decisions", ("decision_id",)) == before
    assert len(_rows(db_path, "idempotency_requests", ("key_digest",))) == 1


def test_replay_preserves_batch_order(repository) -> None:
    digest = key_digest(KEY_ONE)
    payload = {"items": [{"client_item_id": f"item-{i}", "features": FEATURES} for i in range(3)]}
    fingerprint = request_fingerprint(OPERATION_BATCH, payload)
    repository.commit_idempotent(digest, OPERATION_BATCH, fingerprint, _records(3))
    hit = repository.commit_idempotent(digest, OPERATION_BATCH, fingerprint, _records(3, offset=5))
    assert hit.status == IDEMPOTENCY_REPLAYED
    assert [record.decision_id for record in hit.records] == list(V1_IDS)


# --- 冲突 ---------------------------------------------------------------


def test_same_key_different_request_conflicts(repository, db_path: Path) -> None:
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(1))
    other = request_fingerprint(OPERATION_SINGLE, single_payload(OTHER_FEATURES))

    assert repository.lookup_idempotent(digest, OPERATION_SINGLE, other).status == IDEMPOTENCY_CONFLICT
    hit = repository.commit_idempotent(digest, OPERATION_SINGLE, other, _records(1, offset=1))
    assert hit.status == IDEMPOTENCY_CONFLICT
    assert hit.records == ()
    assert len(_rows(db_path, "decisions", ("decision_id",))) == 1, "冲突不得留下候选记录"
    assert len(_rows(db_path, "idempotency_results", ("item_index",))) == 1


def test_same_key_different_operation_conflicts(repository, db_path: Path) -> None:
    """单笔与批量不得共享命名空间：同 payload 换 operation 必须是冲突。"""
    digest = key_digest(KEY_ONE)
    single = request_fingerprint(OPERATION_SINGLE, single_payload(FEATURES))
    repository.commit_idempotent(digest, OPERATION_SINGLE, single, _records(1))
    hit = repository.commit_idempotent(digest, OPERATION_BATCH, single, _records(2))
    assert hit.status == IDEMPOTENCY_CONFLICT
    assert len(_rows(db_path, "decisions", ("decision_id",))) == 1


def test_batch_size_change_conflicts(repository, db_path: Path) -> None:
    digest = key_digest(KEY_ONE)
    one = {"items": [{"client_item_id": "a", "features": FEATURES}]}
    two = {"items": [{"client_item_id": "a", "features": FEATURES},
                     {"client_item_id": "b", "features": FEATURES}]}
    repository.commit_idempotent(
        digest, OPERATION_BATCH, request_fingerprint(OPERATION_BATCH, one), _records(1)
    )
    hit = repository.commit_idempotent(
        digest, OPERATION_BATCH, request_fingerprint(OPERATION_BATCH, two), _records(2)
    )
    assert hit.status == IDEMPOTENCY_CONFLICT


def test_different_keys_with_the_same_request_both_create(repository, db_path: Path) -> None:
    for key, index in ((KEY_TWO, 0), (KEY_THREE, 1)):
        digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, key)
        hit = repository.commit_idempotent(
            digest, OPERATION_SINGLE, fingerprint, [build_record(V1_IDS[index])]
        )
        assert hit.status == IDEMPOTENCY_CREATED
    assert len(_rows(db_path, "decisions", ("decision_id",))) == 2
    assert len(_rows(db_path, "idempotency_requests", ("key_digest",))) == 2


# --- 原子性 -------------------------------------------------------------


def test_failure_while_writing_results_rolls_back_everything(
    repository, db_path: Path, monkeypatch
) -> None:
    from ai_detector.storage import sqlite as sqlite_module

    monkeypatch.setattr(
        sqlite_module,
        "_INSERT_IDEMPOTENCY_RESULT_SQL",
        "INSERT INTO idempotency_results (key_digest, no_such_column) VALUES (?, ?)",
    )
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    with pytest.raises(Exception):
        repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(1))
    assert _rows(db_path, "decisions", ("decision_id",)) == [], "决策也必须一起回滚"
    assert _rows(db_path, "idempotency_requests", ("key_digest",)) == []


def test_failure_while_writing_the_request_row_rolls_back_decisions(
    repository, db_path: Path, monkeypatch
) -> None:
    from ai_detector.storage import sqlite as sqlite_module

    monkeypatch.setattr(
        sqlite_module,
        "_INSERT_IDEMPOTENCY_REQUEST_SQL",
        "INSERT INTO idempotency_requests (key_digest, no_such_column) VALUES (?, ?)",
    )
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    with pytest.raises(Exception):
        repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(2))
    assert _rows(db_path, "decisions", ("decision_id",)) == []
    assert _rows(db_path, "idempotency_requests", ("key_digest",)) == []


def test_a_legitimate_retry_after_a_failure_succeeds(repository, db_path: Path, monkeypatch) -> None:
    """失败只是失败：回滚干净之后，同一次提交重试必须能正常创建。"""
    from ai_detector.storage import sqlite as sqlite_module

    good = sqlite_module._INSERT_IDEMPOTENCY_RESULT_SQL
    monkeypatch.setattr(sqlite_module, "_INSERT_IDEMPOTENCY_RESULT_SQL", "SELECT no_such_column")
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, KEY_ONE)
    with pytest.raises(Exception):
        repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(1))

    monkeypatch.setattr(sqlite_module, "_INSERT_IDEMPOTENCY_RESULT_SQL", good)
    hit = repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(1))
    assert hit.status == IDEMPOTENCY_CREATED
    assert len(_rows(db_path, "decisions", ("decision_id",))) == 1


# --- 损坏映射 -----------------------------------------------------------


def _commit_one(repository, key: str = KEY_ONE) -> tuple[str, str]:
    digest, fingerprint, _ = single_key_and_fingerprint(FEATURES, key)
    repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(1))
    return digest, fingerprint


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE idempotency_requests SET idempotency_schema_version = 'cw009_idempotency_v9'",
        "UPDATE idempotency_requests SET fingerprint_version = 'cw009_fingerprint_v9'",
        "UPDATE idempotency_requests SET operation = 'decision_bulk'",
        "UPDATE idempotency_requests SET request_fingerprint = 'zz'",
        "UPDATE idempotency_requests SET result_count = 2",
        "UPDATE idempotency_results SET item_index = 5",
        "UPDATE idempotency_results SET decision_id = 'not-a-uuid'",
    ],
)
def test_corrupt_mapping_is_reported_as_a_contract_error(
    repository, db_path: Path, statement: str
) -> None:
    digest, fingerprint = _commit_one(repository)
    tamper(db_path, statement)
    with pytest.raises(IdempotencyContractError):
        repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint)


def test_missing_decision_row_is_a_contract_error(repository, db_path: Path) -> None:
    """关联断了（结果行指向的决策不存在）必须报契约错误，而不是返回残缺结果。"""
    digest, fingerprint = _commit_one(repository)
    tamper(db_path, "PRAGMA foreign_keys = OFF", "DELETE FROM decisions")
    with pytest.raises(IdempotencyContractError):
        repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE decisions SET risk_level = 'not_a_level'",
        "UPDATE decisions SET flagged = 1",                 # 与 score/threshold 矛盾
        "UPDATE decisions SET matched_rule_ids = 'not json'",
        "UPDATE decisions SET created_at = '2026-09-25 04:05:06'",
    ],
    ids=["bad_level", "flag_contradiction", "bad_json", "bad_time"],
)
def test_corrupt_stored_decision_is_a_contract_error(
    repository, db_path: Path, statement: str
) -> None:
    """决策行自己坏了：必须抛**幂等契约**错误，而不是让记录层的异常冒到路由被折成 internal_error。"""
    digest, fingerprint = _commit_one(repository)
    tamper(db_path, statement)
    with pytest.raises(IdempotencyContractError):
        repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint)


def test_missing_result_rows_are_a_contract_error(repository, db_path: Path) -> None:
    digest, fingerprint = _commit_one(repository)
    tamper(db_path, "DELETE FROM idempotency_results")
    with pytest.raises(IdempotencyContractError):
        repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint)


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM idempotency_results",
        "UPDATE idempotency_results SET item_index = 7",
        "UPDATE idempotency_results SET decision_id = 'not-a-uuid'",
        "PRAGMA foreign_keys = OFF",   # 下一行才真正断开关联
    ],
    ids=["missing_result", "index_gap", "bad_uuid", "fk_off"],
)
def test_conflict_does_not_mask_a_corrupt_mapping(repository, db_path: Path, statement: str) -> None:
    """**完整校验先于 replay/conflict 判定**：映射坏了就是 500，不能因为这次请求是异请求而报 409。

    否则服务端的存储损坏会被伪装成调用方的冲突——错误分类是契约的一部分。
    """
    digest, _ = _commit_one(repository)
    tamper(db_path, statement, *(["DELETE FROM decisions"] if statement.startswith("PRAGMA") else []))
    other = request_fingerprint(OPERATION_SINGLE, single_payload(OTHER_FEATURES))
    with pytest.raises(IdempotencyContractError):
        repository.lookup_idempotent(digest, OPERATION_SINGLE, other)


def test_corrupt_result_row_also_breaks_the_replay_path(repository, db_path: Path) -> None:
    """同请求（replay）与异请求（conflict）在损坏面前必须**一样**失败。"""
    digest, fingerprint = _commit_one(repository)
    tamper(db_path, "DELETE FROM idempotency_results")
    with pytest.raises(IdempotencyContractError):
        repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint)
    with pytest.raises(IdempotencyContractError):
        repository.commit_idempotent(digest, OPERATION_SINGLE, fingerprint, _records(1))


def test_repository_after_replay_is_still_readable_and_writable(repository, db_path: Path) -> None:
    """重放并开启新连接后，库仍满足完整 v2 契约（重放没有改坏任何结构）。"""
    digest, fingerprint = _commit_one(repository)
    repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint)
    repository.close()
    assert user_version(db_path) == 2
    reopened = open_repository(db_path)
    try:
        assert reopened.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint).status == "replay"
    finally:
        reopened.close()
