"""CW-008 进程内 HTTP smoke：真实 bundle + 冻结政策 + **一次性 v1 SQLite**，走完幂等七步。

用法（项目根目录下执行）::

    python scripts/smoke_idempotency_service.py --config configs/service.yaml

它回答的是"**v1→v2 迁移 + 请求幂等链路通不通、重放是否真的不做事、冲突是否安全**"，
不是"模型准不准"。因此：

- 输入是按 bundle schema 生成的**合成常量特征**，不读取任何真实 CSV、标签或 test 数据；
- 数据库是**本次运行专用**的一个文件，先建成严格的 ``cw007_storage_v1`` 并种入一条**由真实
  政策与真实模型求值产生**的决策，再让服务把它迁移到 v2；运行结束（无论成败）都会删除它
  及其 sidecar；
- 报告只写步骤、状态码、布尔一致性、计数、版本、耗时与环境；**不写**幂等键、摘要、指纹、
  request/client/decision ID、分数、规则 ID、features、特征取值、数据库绝对路径或凭据。

七步：启动并迁移 → keyed 单笔首次/重建 app 后重放 → 同键异请求 409 → keyed 批量首次/重放 →
无 key 两次产生不同 ID → 旧 risk 与 decision GET 契约不变 → 清理本次专用数据库。

产物：``artifacts/reports/cw008_idempotency/smoke_report.json``（默认被 Git 忽略）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector import __version__  # noqa: E402
from ai_detector.rules.config import FROZEN_RULES  # noqa: E402
from ai_detector.rules.engine import fuse  # noqa: E402
from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.config import DEFAULT_CONFIG_PATH, ServiceConfig  # noqa: E402
from ai_detector.service.config import ServiceConfigError  # noqa: E402
from ai_detector.service.runtime import load_policy_engine, load_runtime  # noqa: E402
from ai_detector.storage.config import (  # noqa: E402
    LEGACY_STORAGE_SCHEMA_VERSION,
    LEGACY_USER_VERSION,
    RECORD_SCHEMA_VERSION,
    STORAGE_DIR,
    STORAGE_SCHEMA_VERSION,
    STORAGE_SUFFIX,
    USER_VERSION,
)
from ai_detector.storage.idempotency import (  # noqa: E402
    IDEMPOTENCY_KEY_HEADER,
    REPLAYED_HEADER,
)
from ai_detector.storage.records import StoredDecision  # noqa: E402
from ai_detector.storage.sqlite import (  # noqa: E402
    CANONICAL_DECISIONS_DDL,
    CANONICAL_METADATA_DDL,
    DECISION_COLUMNS,
)

#: 合成常量：数值列给 0.0，类别列给一个固定的占位字符串。它们与任何真实交易无关。
SMOKE_NUMERIC_VALUE = 0.0
SMOKE_CATEGORICAL_VALUE = "smoke_constant"

#: 让第一条规则确定性命中的合成常量（远高于冻结阈值）。只写在脚本里，不写进报告。
SMOKE_RULE_HIT_VALUE = 9_999.0

DEFAULT_OUTPUT = "artifacts/reports/cw008_idempotency/smoke_report.json"

#: 旧 ``/v1/risk/evaluate`` 的成功响应字段集合（CW-006 契约，逐字比对）。
LEGACY_RISK_FIELDS = {
    "request_id",
    "model_version",
    "score",
    "score_semantics",
    "threshold",
    "threshold_policy",
    "flagged",
    "policy_version",
    "policy_sha256",
    "policy_semantics",
    "rule_score",
    "matched_rule_ids",
    "unevaluable_rule_ids",
    "rule_evaluation_complete",
    "signal_points",
    "risk_level",
}

#: 幂等记录涉及的两张表（用于"行数不变"的计数断言）。
MAPPING_TABLES = ("idempotency_requests", "idempotency_results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用真实 bundle 与冻结政策在进程内跑 v1→v2 迁移与幂等七步 smoke 并写出报告。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help=f"服务配置路径（相对项目根目录），默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"报告输出路径（相对项目根目录），默认 {DEFAULT_OUTPUT}",
    )
    return parser


def resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def run_database_path(config: ServiceConfig, run_id: str) -> Path:
    """本次运行专用的数据库路径（与配置同目录，名字带一次性后缀）。"""
    assert config.storage_path is not None
    return config.storage_path.parent / f"{config.storage_path.stem}_smoke_{run_id}{STORAGE_SUFFIX}"


def _rule_features() -> tuple[str, ...]:
    """冻结规则读取的源特征名（用于构造合成输入；不读政策文件）。"""
    return tuple(rule.feature for rule in FROZEN_RULES)


def synthetic_features(runtime: Any, *, hit_first_rule: bool) -> dict[str, Any]:
    """按 bundle 记录的源特征顺序生成一份合成常量特征。"""
    numeric = set(getattr(runtime, "numeric_features", []) or [])
    features: dict[str, Any] = {
        name: SMOKE_NUMERIC_VALUE if name in numeric else SMOKE_CATEGORICAL_VALUE
        for name in runtime.source_features
    }
    for index, feature in enumerate(_rule_features()):
        if feature in features:
            features[feature] = SMOKE_RULE_HIT_VALUE if (hit_first_rule and index == 0) else 0.0
    return features


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_v1_database(
    path: Path, *, engine: Any, runtime: Any, features: dict[str, Any]
) -> str:
    """建一个严格 ``cw007_storage_v1`` 库，并种入一条**真实求值产生**的决策记录。

    这条记录走的是真实的 ``evaluate_rules`` + ``score_many`` + ``fuse``，因此它满足冻结的记录
    契约；迁移测试关心的正是"这样一条真记录能不能原样活到 v2"。
    """
    rule_result = engine.evaluate_rules(features)
    score = float(runtime.score_many([features])[0])
    flagged = bool(score >= runtime.threshold)
    assessment = fuse(rule_result, flagged=flagged)
    record = StoredDecision.from_assessment(
        assessment,
        flagged=flagged,
        decision_id=str(uuid.uuid4()),
        created_at=_utc_now(),
        evaluation_request_id="smoke-seed",
        client_item_id=None,
        model_version=runtime.model_version,
        score=score,
        threshold=runtime.threshold,
        policy_version=engine.policy_version,
        policy_sha256=engine.policy_sha256,
    )

    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute(CANONICAL_METADATA_DDL)
        connection.execute(CANONICAL_DECISIONS_DDL)
        connection.execute(
            "INSERT INTO storage_metadata (key, value) VALUES (?, ?)",
            ("schema_version", LEGACY_STORAGE_SCHEMA_VERSION),
        )
        row = record.to_row()
        connection.execute(
            f"INSERT INTO decisions ({', '.join(DECISION_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(DECISION_COLUMNS))})",
            tuple(row[column] for column in DECISION_COLUMNS),
        )
        connection.execute(f"PRAGMA user_version = {LEGACY_USER_VERSION}")
    finally:
        connection.close()
    return record.decision_id


def read_version(path: Path) -> dict[str, Any]:
    """读回版本与表集合（只读，不写）。"""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = sorted(
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        )
        metadata = str(
            connection.execute(
                "SELECT value FROM storage_metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
        )
        return {
            "user_version": int(connection.execute("PRAGMA user_version").fetchone()[0]),
            "schema_metadata_matches_frozen": metadata == STORAGE_SCHEMA_VERSION,
            "table_count": len(tables),
            "integrity_ok": str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            == "ok",
        }
    finally:
        connection.close()


def count_rows(path: Path, table: str) -> int:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def cleanup_database(path: Path) -> bool:
    """只删除**本次运行自己创建**且**位于安全目录**下的数据库与其 sidecar。"""
    safe_root = (PROJECT_ROOT / STORAGE_DIR).resolve()
    resolved = path.resolve()
    if resolved.parent != safe_root:
        return False
    if not resolved.name.endswith(STORAGE_SUFFIX):
        return False
    removed = False
    for suffix in ("", "-journal", "-wal", "-shm"):
        candidate = Path(str(resolved) + suffix)
        try:
            if candidate.is_file():
                candidate.unlink()
                removed = True
        except OSError:  # pragma: no cover - 仍被占用时留给人工清理
            return False
    return removed


def _safe_body(response: Any) -> Any:
    """只取状态无关的 JSON body；解析失败时返回空字典。"""
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - smoke 不因解析失败而崩
        return {}


def _step(name: str, method: str, route: str, began: float, response: Any, **extra: Any) -> dict:
    body = _safe_body(response)
    entry: dict[str, Any] = {
        "name": name,
        "method": method,
        "route": route,
        "status_code": response.status_code,
        "duration_ms": round((time.perf_counter() - began) * 1000, 3),
        "response_fields": sorted(body) if isinstance(body, dict) else [],
    }
    entry.update(extra)
    return entry


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = ServiceConfig.from_yaml(args.config)
    except ServiceConfigError as error:
        print(f"服务配置加载失败：{error}", file=sys.stderr)
        return 2
    if config.storage_path is None:
        print("服务配置未声明 storage 段，无法执行幂等 smoke。", file=sys.stderr)
        return 4

    try:
        probe_runtime = load_runtime(config)
        probe_engine = load_policy_engine(config)
    except Exception as error:  # noqa: BLE001 - smoke 需要显式失败而不是静默跳过
        print(f"模型或政策加载失败，无法执行 smoke：{type(error).__name__}", file=sys.stderr)
        return 3

    run_id = uuid.uuid4().hex[:12]
    run_db = run_database_path(config, run_id)
    run_config = replace(
        config,
        storage_path=run_db,
        storage_relative_path=run_db.relative_to(PROJECT_ROOT).as_posix(),
    )
    features = synthetic_features(probe_runtime, hit_first_rule=True)
    other_features = synthetic_features(probe_runtime, hit_first_rule=False)
    batch_items = [
        {"client_item_id": "smoke-1", "features": features},
        {"client_item_id": "smoke-2", "features": other_features},
    ]
    # 单笔与批量各用**自己的** key：同一个 key 跨 operation 复用是 409，不是重放。
    headers = {IDEMPOTENCY_KEY_HEADER: f"smoke-run-{run_id}-0001"}
    batch_headers = {IDEMPOTENCY_KEY_HEADER: f"smoke-run-{run_id}-0002"}

    from fastapi.testclient import TestClient

    steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    seeded_id = seed_v1_database(
        run_db, engine=probe_engine, runtime=probe_runtime, features=features
    )
    seeded_rows_before = count_rows(run_db, "decisions")

    single_first: dict[str, Any] | None = None
    batch_first: list[dict[str, Any]] = []
    no_key_ids: list[str] = []

    # --- 1) 启动：v1 必须被迁移，且种子决策仍可读 ---
    app = create_app(run_config)
    with TestClient(app) as client:
        began = time.perf_counter()
        live = client.get("/health/live")
        steps.append(_step("live", "GET", "/health/live", began, live))

        began = time.perf_counter()
        ready = client.get("/health/ready")
        steps.append(_step("ready", "GET", "/health/ready", began, ready))

        began = time.perf_counter()
        seeded = client.get(f"/v1/decisions/{seeded_id}")
        seeded_body = _safe_body(seeded)
        steps.append(
            _step(
                "seeded_decision_after_migration",
                "GET",
                "/v1/decisions/{decision_id}",
                began,
                seeded,
                decision_fields=sorted(seeded_body.get("decision", {}) or {}),
            )
        )
        seeded_readable = (
            seeded.status_code == 200
            and (seeded_body.get("decision", {}) or {}).get("record_schema_version")
            == RECORD_SCHEMA_VERSION
        )

        # --- 2) keyed 单笔首次 ---
        began = time.perf_counter()
        created = client.post(
            "/v1/decisions/evaluate", json={"features": features}, headers=headers
        )
        created_body = _safe_body(created)
        single_first = created_body.get("decision") if isinstance(created_body, dict) else None
        steps.append(
            _step(
                "keyed_single_first",
                "POST",
                "/v1/decisions/evaluate",
                began,
                created,
                replayed_header=created.headers.get(REPLAYED_HEADER),
            )
        )

        # --- 6a) 旧 risk 契约 ---
        began = time.perf_counter()
        legacy = client.post("/v1/risk/evaluate", json={"features": features})
        legacy_body = _safe_body(legacy)
        steps.append(
            _step(
                "legacy_risk",
                "POST",
                "/v1/risk/evaluate",
                began,
                legacy,
                legacy_fields_exact=set(legacy_body) == LEGACY_RISK_FIELDS,
                has_persistence_fields=bool(
                    {"decision_id", "created_at", "record_schema_version"} & set(legacy_body)
                ),
            )
        )
    # app 生命周期结束：连接已关闭

    version_after_migration = read_version(run_db)
    steps.append(
        {
            "name": "schema_after_migration",
            "method": "GET",
            "route": "storage",
            "status_code": None,
            "duration_ms": 0.0,
            "response_fields": [],
            **version_after_migration,
            "user_version_is_current": version_after_migration["user_version"] == USER_VERSION,
        }
    )

    # --- 2b) 重建 app，同请求重放 ---
    rebuilt = create_app(run_config)
    with TestClient(rebuilt) as client:
        began = time.perf_counter()
        replayed = client.post(
            "/v1/decisions/evaluate", json={"features": features}, headers=headers
        )
        replayed_body = _safe_body(replayed)
        replayed_record = replayed_body.get("decision") if isinstance(replayed_body, dict) else None
        steps.append(
            _step(
                "keyed_single_replay_after_restart",
                "POST",
                "/v1/decisions/evaluate",
                began,
                replayed,
                replayed_header=replayed.headers.get(REPLAYED_HEADER),
                field_by_field_identical=replayed_record == single_first,
            )
        )
        single_replay_identical = replayed_record == single_first

        # --- 3) 同键异请求：409 且零新增 ---
        rows_before_conflict = {table: count_rows(run_db, table) for table in MAPPING_TABLES}
        began = time.perf_counter()
        conflict = client.post(
            "/v1/decisions/evaluate", json={"features": other_features}, headers=headers
        )
        conflict_body = _safe_body(conflict)
        steps.append(
            _step(
                "keyed_single_conflict",
                "POST",
                "/v1/decisions/evaluate",
                began,
                conflict,
                error_code=(conflict_body.get("error", {}) or {}).get("code"),
            )
        )
        rows_after_conflict = {table: count_rows(run_db, table) for table in MAPPING_TABLES}
        conflict_added_nothing = rows_before_conflict == rows_after_conflict

        # --- 4) keyed 批量首次与重放 ---
        began = time.perf_counter()
        batch_created = client.post(
            "/v1/decisions/evaluate/batch", json={"items": batch_items}, headers=batch_headers
        )
        batch_body = _safe_body(batch_created)
        batch_first = batch_body.get("decisions", []) if isinstance(batch_body, dict) else []
        steps.append(
            _step(
                "keyed_batch_first",
                "POST",
                "/v1/decisions/evaluate/batch",
                began,
                batch_created,
                replayed_header=batch_created.headers.get(REPLAYED_HEADER),
                item_count=len(batch_first),
            )
        )

        began = time.perf_counter()
        batch_replay = client.post(
            "/v1/decisions/evaluate/batch", json={"items": batch_items}, headers=batch_headers
        )
        batch_replay_body = _safe_body(batch_replay)
        batch_replayed = (
            batch_replay_body.get("decisions", []) if isinstance(batch_replay_body, dict) else []
        )
        order_ok = [item.get("client_item_id") for item in batch_replayed] == [
            item["client_item_id"] for item in batch_items
        ]
        steps.append(
            _step(
                "keyed_batch_replay",
                "POST",
                "/v1/decisions/evaluate/batch",
                began,
                batch_replay,
                replayed_header=batch_replay.headers.get(REPLAYED_HEADER),
                order_preserved=order_ok,
                field_by_field_identical=batch_replayed == batch_first,
            )
        )
        batch_replay_identical = batch_replayed == batch_first and order_ok

        # --- 5) 无 key：两次必须产生不同 ID ---
        began = time.perf_counter()
        plain_a = client.post("/v1/decisions/evaluate", json={"features": features})
        plain_b = client.post("/v1/decisions/evaluate", json={"features": features})
        plain_a_body = _safe_body(plain_a)
        plain_b_body = _safe_body(plain_b)
        no_key_ids = [
            (plain_a_body.get("decision", {}) or {}).get("decision_id"),
            (plain_b_body.get("decision", {}) or {}).get("decision_id"),
        ]
        steps.append(
            _step(
                "no_key_repeat",
                "POST",
                "/v1/decisions/evaluate",
                began,
                plain_b,
                distinct_decisions=bool(no_key_ids[0]) and no_key_ids[0] != no_key_ids[1],
                replayed_header_present=REPLAYED_HEADER in plain_b.headers,
            )
        )

        # --- 6b) decision GET 契约不变 ---
        began = time.perf_counter()
        fetched = client.get(f"/v1/decisions/{no_key_ids[0]}") if no_key_ids[0] else None
        fetched_body = _safe_body(fetched) if fetched is not None else {}
        steps.append(
            _step(
                "decision_get",
                "GET",
                "/v1/decisions/{decision_id}",
                began,
                fetched,
                field_by_field_identical=fetched_body.get("decision")
                == (plain_a_body.get("decision") or None),
            )
        )

    total_ms = round((time.perf_counter() - started) * 1000, 3)
    # 必须在清理**之前**把计数读出来：清理之后文件已经不存在了。
    mapping_rows_after = {table: count_rows(run_db, table) for table in MAPPING_TABLES}
    final_decision_rows = count_rows(run_db, "decisions")

    # --- 7) 清理本次专用数据库 ---
    removed = cleanup_database(run_db)
    sidecars_gone = all(
        not Path(str(run_db) + suffix).exists() for suffix in ("", "-journal", "-wal", "-shm")
    )
    steps.append(
        {
            "name": "cleanup",
            "method": "DELETE",
            "route": "storage",
            "status_code": None,
            "duration_ms": round((time.perf_counter() - started) * 1000 - total_ms, 3),
            "run_scoped_database_removed": removed,
            "no_residual_files": sidecars_gone,
        }
    )

    by_name = {str(step["name"]): step for step in steps}
    consistency = {
        "migrated_to_current_version": version_after_migration["user_version"] == USER_VERSION,
        "schema_metadata_matches_frozen": version_after_migration["schema_metadata_matches_frozen"],
        "seeded_decision_readable_after_migration": seeded_readable,
        # 1 条种子 + keyed 单笔 1 + keyed 批量 2 + 无 key 两次 2 = 6；
        # 两个重放与一次冲突都必须**零新增**，因此总数正好等于这个算式。
        "seed_row_preserved": seeded_rows_before == 1 and final_decision_rows == 6,
        "first_submission_not_marked_replayed": by_name["keyed_single_first"].get(
            "replayed_header"
        )
        == "false",
        "replay_marked_replayed": by_name["keyed_single_replay_after_restart"].get(
            "replayed_header"
        )
        == "true",
        "replay_identical_after_restart": bool(single_replay_identical),
        "conflict_is_409": by_name["keyed_single_conflict"]["status_code"] == 409,
        "conflict_added_no_rows": bool(conflict_added_nothing),
        "batch_first_not_replayed": by_name["keyed_batch_first"].get("replayed_header") == "false",
        "batch_replay_marked_replayed": by_name["keyed_batch_replay"].get("replayed_header")
        == "true",
        "batch_replay_identical_and_ordered": bool(batch_replay_identical),
        "no_key_repeat_creates_distinct_decisions": bool(
            by_name["no_key_repeat"].get("distinct_decisions")
        ),
        "no_key_has_no_replayed_header": not by_name["no_key_repeat"].get(
            "replayed_header_present"
        ),
        "legacy_risk_contract_unchanged": bool(by_name["legacy_risk"].get("legacy_fields_exact")),
        "legacy_risk_has_no_persistence_fields": not by_name["legacy_risk"].get(
            "has_persistence_fields"
        ),
        "decision_get_identical": bool(by_name["decision_get"].get("field_by_field_identical")),
        # 一个 key 的单笔映射（1 条结果）+ 一个 key 的批量映射（2 条结果）＝ 2 + 3。
        "mapping_rows_exactly_as_expected": mapping_rows_after
        == {"idempotency_requests": 2, "idempotency_results": 3},
    }

    report: dict[str, Any] = {
        "kind": "cw008_idempotency_smoke",
        "work_package": "CW-008",
        "generated_at": _utc_now(),
        "package_version": __version__,
        "service": {
            "name": config.name,
            "api_version": config.api_version,
            "batch_limit": config.batch_limit,
        },
        "storage": {
            "current_schema_version": STORAGE_SCHEMA_VERSION,
            "legacy_schema_version": LEGACY_STORAGE_SCHEMA_VERSION,
            "user_version": USER_VERSION,
            "legacy_user_version": LEGACY_USER_VERSION,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "database_location": STORAGE_DIR,
            "run_scoped_database": True,
            "note": "本次运行先建 v1 并种入一条真实求值记录，再让服务迁移到 v2；结束后删除该文件。",
        },
        "model": {"model_version": config.model_version, "source_features": config.expected_source_features},
        "policy": {"policy_version": probe_engine.policy_version},
        "smoke_input": {
            "kind": "synthetic_constants",
            "note": "按 bundle schema 生成的合成常量特征；不含任何真实交易数据。",
            "single_items": 1,
            "batch_items": len(batch_items),
        },
        "steps": steps,
        "consistency": consistency,
        "cleanup": {
            "run_scoped_database_removed": removed,
            "no_residual_files": sidecars_gone,
        },
        "all_steps_status_ok": all(
            step["status_code"] in (200, 201, 409) for step in steps if step["status_code"] is not None
        ),
        "total_duration_ms": total_ms,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "sqlite": sqlite3.sqlite_version,
        },
        "limitations": [
            "记录只保存派生结果（版本、分数、规则求值事实、研究型等级），不保存 features、请求体、"
            "header、原始 path/query 或任何输入载荷；幂等也只存两个摘要与有序 decision ID 关联。",
            "字段是研究型信号，不是概率，也不是业务规则或任何生产处置。",
            "本包明确**不是 exactly-once**：同键并发可能让模型执行多次，只保证最终只提交一组结果。",
            "smoke 只证明迁移与幂等链路自洽、可读回、可重放；它不是模型效果或并发压测结论。",
            "单机 SQLite，无 TLS、无认证、默认仅监听本机地址。",
            "不提供 TTL、列表查询、更新删除、通用迁移、PostgreSQL、Docker 或 CI。",
        ],
    }
    report["all_steps_ok"] = report["all_steps_status_ok"] and all(consistency.values())
    output = resolve_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for step in steps:
        status = "" if step["status_code"] is None else step["status_code"]
        print(
            f"  {step['name']:<36} {step['method']:<6} {step['route']:<32} -> {status}"
        )
    print(f"smoke 全通过：{report['all_steps_ok']} | 总耗时 {total_ms} ms")
    print(f"报告：{output.relative_to(PROJECT_ROOT).as_posix()}")
    print("提醒：smoke 只证明链路自洽性，不是模型效果指标，也不是并发结论。")
    return 0 if report["all_steps_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
