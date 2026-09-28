"""CW-007 进程内 HTTP smoke：真实 bundle + 冻结政策 + 一次性 SQLite，走完六步持久化链路。

用法（项目根目录下执行）::

    python scripts/smoke_persistence_service.py --config configs/service.yaml

它回答的问题是"**持久化链路通不通、读写是否自洽、重启后还在不在、事务是否成批**"，
不是"模型准不准"。因此：

- 输入是按 bundle schema 生成的**合成常量特征**，不读取任何真实 CSV、标签或 test 数据；
- 数据库是**本次运行专用**的一个文件，位于配置允许的 ``artifacts/runtime/`` 下，
  运行结束（无论成败）都会被删除，包括 journal 等 sidecar；
- 报告不写入 decision ID、request/client ID、分数、规则 ID、记录内容、features、
  特征取值、数据库绝对路径或凭据。

六步：live/ready → 单笔持久化并读回 → **重建 app** 后仍可读 → 两笔批量与逐条查询 →
旧 ``/v1/risk/evaluate`` 契约不变 → 关闭连接并清理本次专用数据库。

产物：``artifacts/reports/cw007_persistence/smoke_report.json``（默认被 Git 忽略）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector import __version__  # noqa: E402
from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.config import DEFAULT_CONFIG_PATH, ServiceConfig  # noqa: E402
from ai_detector.service.config import ServiceConfigError  # noqa: E402
from ai_detector.service.runtime import load_policy_engine, load_runtime  # noqa: E402
from ai_detector.storage.config import (  # noqa: E402
    RECORD_SCHEMA_VERSION,
    STORAGE_DIR,
    STORAGE_SCHEMA_VERSION,
    STORAGE_SUFFIX,
)

#: 合成常量：数值列给 0.0，类别列给一个固定的占位字符串。它们与任何真实交易无关。
SMOKE_NUMERIC_VALUE = 0.0
SMOKE_CATEGORICAL_VALUE = "smoke_constant"

#: 让第一条规则确定性命中的合成常量（远高于冻结阈值）。取值写在脚本里，不写进报告。
SMOKE_RULE_HIT_VALUE = 9_999.0

DEFAULT_OUTPUT = "artifacts/reports/cw007_persistence/smoke_report.json"

#: 旧 ``/v1/risk/evaluate`` 的成功响应字段集合（CW-006 契约，逐字比对）。
RISK_RESPONSE_FIELDS = frozenset(
    {
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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用真实 bundle 与冻结政策在进程内跑持久化六步 smoke 并写出报告。"
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
    from ai_detector.rules.config import FROZEN_RULES

    return tuple(rule.feature for rule in FROZEN_RULES)


def synthetic_features(runtime: Any, *, hit_first_rule: bool) -> dict[str, Any]:
    """按 bundle 记录的源特征顺序生成一份合成常量特征。

    ``hit_first_rule=True`` 时把第一条规则的源特征设为确定性命中的合成常量、其余规则字段
    设为不命中；``False`` 时全部不命中。取值都是脚本内写死的常量，与任何真实交易无关。
    """
    numeric = set(getattr(runtime, "numeric_features", []) or [])
    features: dict[str, Any] = {
        name: SMOKE_NUMERIC_VALUE if name in numeric else SMOKE_CATEGORICAL_VALUE
        for name in runtime.source_features
    }
    for index, feature in enumerate(_rule_features()):
        if feature in features:
            features[feature] = SMOKE_RULE_HIT_VALUE if (hit_first_rule and index == 0) else 0.0
    return features


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = ServiceConfig.from_yaml(args.config)
    except ServiceConfigError as error:
        print(f"服务配置加载失败：{error}", file=sys.stderr)
        return 2
    if config.storage_path is None:
        print("服务配置未声明 storage 段，无法执行持久化 smoke。", file=sys.stderr)
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
    features_no_hit = synthetic_features(probe_runtime, hit_first_rule=False)
    batch_items = [
        {"client_item_id": "smoke-1", "features": features},
        {"client_item_id": "smoke-2", "features": features_no_hit},
    ]

    from fastapi.testclient import TestClient

    steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    decision_id: str | None = None
    single_record: dict[str, Any] | None = None
    batch_records: list[dict[str, Any]] = []
    after_restart: dict[str, Any] | None = None

    # --- 1) 启动并检查探针 ---
    app = create_app(run_config)
    with TestClient(app) as client:
        for name, route in (("live", "/health/live"), ("ready", "/health/ready")):
            began = time.perf_counter()
            response = client.get(route)
            body = _safe_body(response)
            steps.append(
                {
                    "name": name,
                    "method": "GET",
                    "route": route,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - began) * 1000, 3),
                    "response_fields": sorted(body) if isinstance(body, dict) else [],
                    "status": body.get("status") if isinstance(body, dict) else None,
                }
            )

        # --- 2) 单笔持久化并读回 ---
        began = time.perf_counter()
        created = client.post("/v1/decisions/evaluate", json={"features": features})
        created_body = _safe_body(created)
        single_record = created_body.get("decision") if isinstance(created_body, dict) else None
        decision_id = (single_record or {}).get("decision_id")
        steps.append(
            {
                "name": "decision_post",
                "method": "POST",
                "route": "/v1/decisions/evaluate",
                "status_code": created.status_code,
                "duration_ms": round((time.perf_counter() - began) * 1000, 3),
                "response_fields": sorted(created_body) if isinstance(created_body, dict) else [],
                "decision_fields": sorted(single_record) if single_record else [],
                "has_decision_id": bool(decision_id),
                "risk_level": (single_record or {}).get("risk_level"),
                "rule_match_count": len((single_record or {}).get("matched_rule_ids", [])),
                "score_finite": isinstance((single_record or {}).get("score"), (int, float)),
            }
        )

        began = time.perf_counter()
        fetched = client.get(f"/v1/decisions/{decision_id}") if decision_id else None
        fetched_body = _safe_body(fetched) if fetched is not None else {}
        fetched_record = fetched_body.get("decision") if isinstance(fetched_body, dict) else None
        steps.append(
            {
                "name": "decision_get",
                "method": "GET",
                "route": "/v1/decisions/{decision_id}",
                "status_code": fetched.status_code if fetched is not None else None,
                "duration_ms": round((time.perf_counter() - began) * 1000, 3),
                "response_fields": sorted(fetched_body) if isinstance(fetched_body, dict) else [],
                "field_by_field_identical": fetched_record == single_record,
            }
        )
    # app 生命周期结束：连接已关闭

    # --- 3) 重建 app，用同一个数据库再读一次 ---
    rebuilt = create_app(run_config)
    with TestClient(rebuilt) as client:
        began = time.perf_counter()
        again = client.get(f"/v1/decisions/{decision_id}") if decision_id else None
        again_body = _safe_body(again) if again is not None else {}
        after_restart = again_body.get("decision") if isinstance(again_body, dict) else None
        steps.append(
            {
                "name": "decision_get_after_restart",
                "method": "GET",
                "route": "/v1/decisions/{decision_id}",
                "status_code": again.status_code if again is not None else None,
                "duration_ms": round((time.perf_counter() - began) * 1000, 3),
                "response_fields": sorted(again_body) if isinstance(again_body, dict) else [],
                "survived_restart": after_restart == single_record,
            }
        )

        # --- 4) 两笔批量 + 逐条查询 ---
        began = time.perf_counter()
        batch = client.post("/v1/decisions/evaluate/batch", json={"items": batch_items})
        batch_body = _safe_body(batch)
        records = batch_body.get("decisions") if isinstance(batch_body, dict) else None
        batch_records = list(records or [])
        order_preserved = [
            record.get("client_item_id") for record in batch_records
        ] == ["smoke-1", "smoke-2"]
        queried_ok = True
        for record in batch_records:
            probe = client.get(f"/v1/decisions/{record.get('decision_id')}")
            probe_body = _safe_body(probe)
            if probe.status_code != 200 or probe_body.get("decision") != record:
                queried_ok = False
        steps.append(
            {
                "name": "decision_batch",
                "method": "POST",
                "route": "/v1/decisions/evaluate/batch",
                "status_code": batch.status_code,
                "duration_ms": round((time.perf_counter() - began) * 1000, 3),
                "response_fields": sorted(batch_body) if isinstance(batch_body, dict) else [],
                "batch_size": (batch_body or {}).get("batch_size"),
                "order_preserved": order_preserved,
                "unique_decision_ids": len({r.get("decision_id") for r in batch_records}),
                "all_queried_records_identical": queried_ok,
            }
        )

        # --- 5) 旧 risk 接口契约不变 ---
        began = time.perf_counter()
        risk = client.post("/v1/risk/evaluate", json={"features": features})
        risk_body = _safe_body(risk)
        risk_fields = set(risk_body) if isinstance(risk_body, dict) else set()
        steps.append(
            {
                "name": "legacy_risk",
                "method": "POST",
                "route": "/v1/risk/evaluate",
                "status_code": risk.status_code,
                "duration_ms": round((time.perf_counter() - began) * 1000, 3),
                "response_fields": sorted(risk_fields),
                "contract_unchanged": risk_fields == RISK_RESPONSE_FIELDS,
                "no_persistence_fields": not (risk_fields & {"decision", "decision_id"}),
            }
        )

    # --- 6) 清理本次专用数据库 ---
    cleaned = cleanup_database(run_db)
    database_gone = not run_db.exists() and not any(
        Path(str(run_db) + suffix).exists() for suffix in ("-journal", "-wal", "-shm")
    )

    total_ms = round((time.perf_counter() - started) * 1000, 3)
    by_name = {step["name"]: step for step in steps}
    report = {
        "work_package": "CW-007",
        "kind": "in_process_http_persistence_smoke",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_version": __version__,
        # 只报告**声明级**服务配置，并刻意剔除 storage_* 两项：本次用的是一次性数据库，
        # 它的文件名既无价值也不该进报告（数据库位置统一由下面 storage 段说明）。
        "service": {
            key: value
            for key, value in config.public_dict().items()
            if not key.startswith("storage_")
        },
        "model": {
            "model_version": probe_runtime.model_version,
            "bundle_sha256": probe_runtime.bundle_sha256,
            "threshold": probe_runtime.threshold,
            "score_semantics": probe_runtime.score_semantics,
            "n_source_features": len(probe_runtime.source_features),
        },
        "policy": {
            "policy_version": probe_engine.policy_version,
            "policy_sha256": probe_engine.policy_sha256,
            "policy_semantics": probe_engine.policy_semantics,
            "n_rules": len(probe_engine.rules),
        },
        "storage": {
            "storage_schema_version": STORAGE_SCHEMA_VERSION,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "database_location": STORAGE_DIR,
            "run_scoped_database": True,
            "note": "本次运行使用一个一次性的数据库文件，运行结束即删除；其文件名不进报告。",
        },
        "smoke_input": {
            "kind": "synthetic_constant",
            "n_items_in_batch": len(batch_items),
            "note": (
                "输入是按 bundle schema 生成的合成常量特征（数值列 0.0、类别列固定占位串），"
                "其中规则字段取脚本内写死的合成常量以确定性触发一条规则。"
                "不读取任何真实 CSV 或 test 数据；报告因此**不是**模型效果证据。"
            ),
        },
        "steps": steps,
        "consistency": {
            "single_record_read_back_identical": by_name["decision_get"].get(
                "field_by_field_identical"
            )
            is True,
            "record_survived_app_restart": by_name["decision_get_after_restart"].get(
                "survived_restart"
            )
            is True,
            "batch_order_preserved": by_name["decision_batch"].get("order_preserved") is True,
            "batch_records_queried_identical": by_name["decision_batch"].get(
                "all_queried_records_identical"
            )
            is True,
            "batch_ids_unique": by_name["decision_batch"].get("unique_decision_ids")
            == len(batch_items),
            "legacy_risk_contract_unchanged": by_name["legacy_risk"].get("contract_unchanged")
            is True,
            "legacy_risk_has_no_persistence_fields": by_name["legacy_risk"].get(
                "no_persistence_fields"
            )
            is True,
        },
        "cleanup": {
            "run_scoped_database_removed": cleaned,
            "no_residual_files": database_gone,
        },
        "total_duration_ms": total_ms,
        "all_steps_ok": False,  # 由下方统一合成（见下）
        "all_steps_status_ok": all(
            step["status_code"] in (200, 201) for step in steps if step["status_code"] is not None
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sqlite": _sqlite_version(),
            "fastapi": _version("fastapi"),
            "pydantic": _version("pydantic"),
            "uvicorn": _version("uvicorn"),
            "httpx": _version("httpx"),
            "lightgbm": _version("lightgbm"),
        },
        "limitations": [
            "记录只保存派生结果（版本、分数、规则求值、研究型等级），不含 features、请求体、"
            "header、原始 path/query 或其它输入载荷。",
            "字段与等级都是研究型信号，不是概率，不是业务规则，也不构成任何生产处置。",
            "本工作包**不做幂等**：相同输入或相同 X-Request-ID 重复提交会产生新的 decision ID "
            "与新记录，不得据此推断 exactly-once。",
            "smoke 只证明持久化链路可用、读写与重启一致、批量成批，不构成模型效果或生产适用性声明。",
            "服务无认证、限流与 TLS，默认仅监听本机回环地址，不得直接暴露到公网。",
            "本工作包不含列表查询、更新删除、迁移、PostgreSQL、Docker 与 CI。",
        ],
    }
    # 全通过 = 所有步骤状态码正常 + 所有一致性检查为真 + 本次专用数据库已彻底清理。
    report["all_steps_ok"] = (
        report["all_steps_status_ok"]
        and all(report["consistency"].values())
        and cleaned
        and database_gone
    )

    output_path = resolve_output(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for step in steps:
        print(
            f"  {step['name']:<30} {step['method']:<5} {step['route']:<32} "
            f"-> {step['status_code']}  {step['duration_ms']} ms"
        )
    print(f"smoke 全通过：{report['all_steps_ok']} | 总耗时 {total_ms} ms")
    print(f"本次专用数据库已清理：{cleaned and database_gone}")
    print(f"报告：{config.relative(output_path)}")
    print("提醒：smoke 只证明持久化链路与自洽性，不是模型效果指标，也不是生产处置依据。")
    return 0 if report["all_steps_ok"] else 1


def _safe_body(response: Any) -> Any:
    if response is None:  # pragma: no cover - 防御性
        return {}
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - 非 JSON 响应（不应出现）按空处理
        return {}


def _sqlite_version() -> str:
    import sqlite3

    return str(sqlite3.sqlite_version)


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:  # pragma: no cover - 依赖缺失已在加载阶段暴露
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())
