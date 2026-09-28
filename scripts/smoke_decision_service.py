"""CW-006 进程内 HTTP smoke：真实 CW-004 bundle + 冻结政策走一遍六步链路。

用法（项目根目录下执行）::

    python scripts/smoke_decision_service.py --config configs/service.yaml

它回答的问题是"**接口与规则链路通不通、政策与响应是否自洽**"，不是"模型准不准"、
也不是"风险等级对不对"：

- 输入是按 bundle schema 生成的**合成常量特征**，不读取任何真实 CSV、标签或 test 数据；
- 报告里的分数只作为接口连通性证据；``risk_level`` 只作为"模型信号 + 规则信号"链路的
  一致性证据，**不得**被读成模型效果或生产适用性结论；
- 报告不写入 199 列载荷、任何特征取值、绝对路径或凭据。

六步：live、ready、policy、旧 score、单笔 risk、两笔 batch risk。

产物：``artifacts/reports/cw006_decision/smoke_report.json``（默认被 Git 忽略）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector import __version__  # noqa: E402
from ai_detector.rules.config import (  # noqa: E402
    FLAGGED_SIGNAL_POINTS,
    FROZEN_BANDS,
    MATCHED_RULE_SIGNAL_POINTS,
    RULE_SCORE_DENOMINATOR,
)
from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.config import DEFAULT_CONFIG_PATH, ServiceConfig  # noqa: E402
from ai_detector.service.config import ServiceConfigError  # noqa: E402
from ai_detector.service.runtime import load_policy_engine, load_runtime  # noqa: E402

#: 合成常量：数值列给 0.0，类别列给一个固定的占位字符串。它们与任何真实交易无关。
SMOKE_NUMERIC_VALUE = 0.0
SMOKE_CATEGORICAL_VALUE = "smoke_constant"

#: 让第一条规则**确定性命中**的合成常量（远高于冻结阈值），以及让第二条**确定性不命中**的值。
#: 具体取值写在脚本里而不是报告里——报告只需说明"用了固定合成常量"。
SMOKE_RULE_HIT_VALUE = 9_999.0
SMOKE_RULE_MISS_VALUE = 0.0

DEFAULT_OUTPUT = "artifacts/reports/cw006_decision/smoke_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用真实 bundle 与冻结政策在进程内跑六个接口的 smoke 并写出报告。"
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
    """把输出路径解析为绝对路径（相对项目根目录）。"""
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def synthetic_features(runtime: Any) -> dict[str, Any]:
    """按 bundle 记录的源特征顺序生成一份合成常量特征。

    只需要列名与"数值/类别"划分，不接触任何真实数据；两个规则字段的取值随后由
    :func:`_apply_rule_constants` 覆盖。
    """
    numeric = set(getattr(runtime, "numeric_features", []) or [])
    features: dict[str, Any] = {}
    for name in runtime.source_features:
        features[name] = SMOKE_NUMERIC_VALUE if name in numeric else SMOKE_CATEGORICAL_VALUE
    return features


def _apply_rule_constants(features: dict[str, Any], rules: Any, *, hit_index: int | None) -> None:
    """按政策声明的源特征写入合成常量（只写规则声明的那两列，其余保持常量）。

    ``hit_index`` 为命中的规则序号；``None`` 表示两条都不命中。
    """
    for index, rule in enumerate(rules):
        target = SMOKE_RULE_HIT_VALUE if index == hit_index else SMOKE_RULE_MISS_VALUE
        if rule.feature in features:
            features[rule.feature] = target


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = ServiceConfig.from_yaml(args.config)
    except ServiceConfigError as error:
        print(f"服务配置加载失败：{error}", file=sys.stderr)
        return 2

    # 先单独加载一次运行时与政策：只为拿到源特征清单与规则声明来构造合成输入。
    # 应用自身在 lifespan 里会再加载一次——两次都由同一套 loader 完成，逻辑一致。
    if config.policy_path is None:
        print("服务配置未声明 policy 段，无法执行 CW-006 smoke。", file=sys.stderr)
        return 4
    try:
        probe_runtime = load_runtime(config)
    except Exception as error:  # noqa: BLE001 - smoke 需要显式失败而不是静默跳过
        print(f"bundle 加载失败，无法执行 smoke：{type(error).__name__}", file=sys.stderr)
        return 3
    try:
        probe_engine = load_policy_engine(config)
    except Exception as error:  # noqa: BLE001
        print(f"政策加载失败，无法执行 smoke：{type(error).__name__}", file=sys.stderr)
        return 5

    features_hit = synthetic_features(probe_runtime)
    _apply_rule_constants(features_hit, probe_engine.rules, hit_index=0)
    features_miss = synthetic_features(probe_runtime)
    _apply_rule_constants(features_miss, probe_engine.rules, hit_index=None)

    batch_items = [
        {"client_item_id": "smoke-1", "features": features_hit},
        {"client_item_id": "smoke-2", "features": features_miss},
    ]

    from fastapi.testclient import TestClient

    app = create_app(config)
    steps: list[dict[str, Any]] = []
    started = time.perf_counter()

    with TestClient(app) as client:
        for name, method, url, payload in (
            ("live", "GET", "/health/live", None),
            ("ready", "GET", "/health/ready", None),
            ("policy", "GET", "/v1/policy", None),
            ("score", "POST", "/v1/score", {"features": features_hit}),
            ("risk", "POST", "/v1/risk/evaluate", {"features": features_hit}),
            ("risk_batch", "POST", "/v1/risk/evaluate/batch", {"items": batch_items}),
        ):
            step_started = time.perf_counter()
            response = (
                client.request(method, url, json=payload)
                if payload
                else client.request(method, url)
            )
            elapsed_ms = round((time.perf_counter() - step_started) * 1000, 3)
            body = _safe_body(response)
            steps.append(
                {
                    "name": name,
                    "method": method,
                    "route": url,
                    "status_code": response.status_code,
                    "duration_ms": elapsed_ms,
                    "response_fields": sorted(body) if isinstance(body, dict) else [],
                    "request_id_echoed": bool(response.headers.get(config.request_id_header)),
                    **_step_extra(name, body, probe_engine),
                }
            )

    total_ms = round((time.perf_counter() - started) * 1000, 3)
    by_name = {step["name"]: step for step in steps}
    report = {
        "work_package": "CW-006",
        "kind": "in_process_http_smoke",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_version": __version__,
        "service": config.public_dict(),
        "model": {
            "model_version": probe_runtime.model_version,
            "bundle_sha256": probe_runtime.bundle_sha256,
            "threshold": probe_runtime.threshold,
            "threshold_policy": probe_runtime.threshold_policy,
            "score_semantics": probe_runtime.score_semantics,
            "n_source_features": len(probe_runtime.source_features),
        },
        "policy": {
            "policy_version": probe_engine.policy_version,
            "policy_schema_version": probe_engine.policy_schema_version,
            "policy_sha256": probe_engine.policy_sha256,
            "policy_semantics": probe_engine.policy_semantics,
            "threshold_source": probe_engine.threshold_source,
            "rule_ids": [rule.rule_id for rule in probe_engine.rules],
            "rule_thresholds": [float(rule.threshold) for rule in probe_engine.rules],
            "operators": [rule.operator for rule in probe_engine.rules],
            "fusion": {
                "flagged_signal_points": FLAGGED_SIGNAL_POINTS,
                "matched_rule_signal_points": MATCHED_RULE_SIGNAL_POINTS,
                "rule_score_denominator": RULE_SCORE_DENOMINATOR,
                "bands": {band.name: [band.low, band.high] for band in FROZEN_BANDS},
            },
            "disclaimer_present": bool(probe_engine.disclaimer),
        },
        "smoke_input": {
            "kind": "synthetic_constant",
            "n_items_in_batch": len(batch_items),
            "note": (
                "输入是按 bundle schema 生成的合成常量特征（数值列 0.0、类别列固定占位串）；"
                "为实现确定性规则命中，两个规则字段被写入固定的合成常量（一个远高于冻结阈值、"
                "一个取 0.0），取值写在脚本源码里而不写进本报告。"
                "不读取任何真实 CSV 或 test 数据；因此本报告**不是**模型效果证据。"
            ),
        },
        "steps": steps,
        "consistency": {
            # 政策指纹：服务配置里固定的那一份，是否就是 /v1/policy 报出的那一份。
            "policy_sha256_matches_config": by_name["policy"].get("policy_sha256")
            == config.policy_sha256,
            # 旧评分接口的阈值与语义在政策启用后是否保持不变。
            "score_semantics_unchanged": by_name["score"].get("score_semantics")
            == probe_runtime.score_semantics,
            "score_threshold_unchanged": by_name["score"].get("threshold")
            == probe_runtime.threshold,
            # 单笔风险等级是否与"flagged + 命中数"的冻结真值表自洽。
            "risk_level_matches_truth_table": by_name["risk"].get("truth_table_consistent")
            is True,
            "batch_order_preserved": by_name["risk_batch"].get("order_preserved") is True,
        },
        "total_duration_ms": total_ms,
        "all_steps_ok": all(step["status_code"] == 200 for step in steps),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "fastapi": _version("fastapi"),
            "pydantic": _version("pydantic"),
            "uvicorn": _version("uvicorn"),
            "httpx": _version("httpx"),
            "lightgbm": _version("lightgbm"),
            "yaml": _version("yaml"),
        },
        "limitations": [
            "模型分数是未校准的模型分数，不是欺诈概率；flagged 只是 score >= 冻结阈值。",
            "risk_level 是研究型信号分级（模型 2 点 + 每条命中规则 1 点），不是概率，"
            "不是业务规则，也不构成任何批准/拒绝等生产处置。",
            "两条规则的阈值来自公开匿名数据训练分区的 0.99 分位统计，C13 是匿名特征，"
            "本项目不为它赋予任何业务含义。",
            "smoke 只证明接口与规则链路可用、政策与响应自洽，不构成模型效果或生产适用性声明。",
            "服务无认证、限流与 TLS，默认仅监听本机回环地址，不得直接暴露到公网。",
            "本工作包不含数据库、结果查询、持久化幂等、动态规则 DSL、Docker 与 CI。",
        ],
    }

    output_path = resolve_output(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for step in steps:
        print(
            f"  {step['name']:<11} {step['method']:<5} {step['route']:<26} "
            f"-> {step['status_code']}  {step['duration_ms']} ms"
        )
    print(f"smoke 全通过：{report['all_steps_ok']} | 总耗时 {total_ms} ms")
    print(f"报告：{config.relative(output_path)}")
    print("提醒：smoke 只证明链路与自洽性，不是模型效果指标，也不是生产处置依据。")
    return 0 if report["all_steps_ok"] else 1


def _step_extra(name: str, body: Any, engine: Any) -> dict[str, Any]:
    """把"结论性字段"提出来，但**不写入任何载荷或特征取值**。"""
    if not isinstance(body, dict):
        return {}
    if name == "live":
        return {"status": body.get("status")}
    if name == "ready":
        return {"status": body.get("status")}
    if name == "policy":
        return {
            "policy_version": body.get("policy_version"),
            "policy_sha256": body.get("policy_sha256"),
            "n_rules": len(body.get("rules", [])),
            "n_risk_levels": len(body.get("risk_levels", [])),
        }
    if name == "score":
        return {
            "score_finite": isinstance(body.get("score"), (int, float)),
            "score_in_unit_interval": _in_unit_interval(body.get("score")),
            "threshold": body.get("threshold"),
            "score_semantics": body.get("score_semantics"),
            "flagged": body.get("flagged"),
        }
    if name == "risk":
        return {
            "score_finite": isinstance(body.get("score"), (int, float)),
            "threshold": body.get("threshold"),
            "flagged": body.get("flagged"),
            "rule_match_count": len(body.get("matched_rule_ids", [])),
            "unevaluable_rule_count": len(body.get("unevaluable_rule_ids", [])),
            "rule_score": body.get("rule_score"),
            "rule_evaluation_complete": body.get("rule_evaluation_complete"),
            "signal_points": body.get("signal_points"),
            "risk_level": body.get("risk_level"),
            "policy_version": body.get("policy_version"),
            "truth_table_consistent": _truth_table_consistent(body),
        }
    if name == "risk_batch":
        results = body.get("results", [])
        ids = [item.get("client_item_id") for item in results]
        return {
            "batch_size": body.get("batch_size"),
            "order_preserved": ids == ["smoke-1", "smoke-2"],
            "n_items_with_complete_rule_evaluation": sum(
                1 for item in results if item.get("rule_evaluation_complete") is True
            ),
            "levels_present": sorted(
                {str(item.get("risk_level")) for item in results}
            ),
            "all_scores_finite": all(
                isinstance(item.get("score"), (int, float)) for item in results
            ),
            "threshold": body.get("threshold"),
            "policy_version": body.get("policy_version"),
        }
    return {}


def _truth_table_consistent(body: dict[str, Any]) -> bool:
    """重算冻结真值表并与响应逐字段比对（**独立于服务实现**的一份校验）。"""
    if body.get("rule_evaluation_complete") is not True:
        return False
    matched = len(body.get("matched_rule_ids", []))
    points = (FLAGGED_SIGNAL_POINTS if body.get("flagged") else 0) + (
        matched * MATCHED_RULE_SIGNAL_POINTS
    )
    expected_level = next(
        (band.name for band in FROZEN_BANDS if band.low <= points <= band.high), None
    )
    return (
        body.get("signal_points") == points
        and body.get("risk_level") == expected_level
        and body.get("rule_score") == matched / RULE_SCORE_DENOMINATOR
    )


def _in_unit_interval(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def _safe_body(response: Any) -> Any:
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - 非 JSON 响应（不应出现）按空处理
        return {}


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:  # pragma: no cover - 依赖缺失已在加载阶段暴露
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())
