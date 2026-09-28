"""进程内 HTTP smoke：用真实 CW-004 bundle 走一遍五个接口并写出报告。

用法（项目根目录下执行）::

    python scripts/smoke_service.py --config configs/service.yaml

它回答的问题是"**接口通不通**"，不是"模型准不准"：

- 输入是按 bundle schema 生成的**合成常量特征**，不读取任何真实 CSV 或 test 数据；
- 报告里的分数只是接口连通性证据，**不是**模型效果指标；
- 报告不写入 199 列载荷、特征取值、类别取值、绝对路径或凭据。

产物：``artifacts/reports/cw005_service/smoke_report.json``（默认被 Git 忽略）。
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
from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.config import DEFAULT_CONFIG_PATH, ServiceConfig  # noqa: E402
from ai_detector.service.config import ServiceConfigError  # noqa: E402
from ai_detector.service.runtime import load_runtime  # noqa: E402

#: 合成常量：数值列给 0.0，类别列给一个固定的占位字符串。
#: 它们与任何真实交易无关，只用于证明接口链路可用。
SMOKE_NUMERIC_VALUE = 0.0
SMOKE_CATEGORICAL_VALUE = "smoke_constant"

DEFAULT_OUTPUT = "artifacts/reports/cw005_service/smoke_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用真实 bundle 在进程内跑五个接口的 smoke 并写出报告。"
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

    只需要列名与"数值/类别"的划分，因此不接触任何真实数据。
    """
    numeric = set(getattr(runtime, "numeric_features", []) or [])
    features: dict[str, Any] = {}
    for name in runtime.source_features:
        features[name] = SMOKE_NUMERIC_VALUE if name in numeric else SMOKE_CATEGORICAL_VALUE
    return features


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = ServiceConfig.from_yaml(args.config)
    except ServiceConfigError as error:
        print(f"服务配置加载失败：{error}", file=sys.stderr)
        return 2

    # 先单独加载一次运行时：只为拿到源特征清单来构造合成输入。
    # 应用自身在 lifespan 里会再加载一次——两次都由同一个 loader 完成，逻辑一致。
    try:
        probe_runtime = load_runtime(config)
    except Exception as error:  # noqa: BLE001 - smoke 需要显式失败而不是静默跳过
        print(f"bundle 加载失败，无法执行 smoke：{type(error).__name__}", file=sys.stderr)
        return 3

    features = synthetic_features(probe_runtime)
    batch_items = [
        {"client_item_id": "smoke-1", "features": features},
        {"client_item_id": "smoke-2", "features": features},
    ]

    from fastapi.testclient import TestClient

    app = create_app(config)
    steps: list[dict[str, Any]] = []
    started = time.perf_counter()

    with TestClient(app) as client:
        for name, method, url, payload in (
            ("live", "GET", "/health/live", None),
            ("ready", "GET", "/health/ready", None),
            ("model", "GET", "/v1/model", None),
            ("score", "POST", "/v1/score", {"features": features}),
            ("score_batch", "POST", "/v1/score/batch", {"items": batch_items}),
        ):
            step_started = time.perf_counter()
            response = client.request(method, url, json=payload) if payload else client.request(method, url)
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
                    "request_id_echoed": bool(
                        response.headers.get(config.request_id_header)
                    ),
                    **_step_extra(name, body, probe_runtime),
                }
            )

    total_ms = round((time.perf_counter() - started) * 1000, 3)
    report = {
        "work_package": "CW-005",
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
        "smoke_input": {
            "kind": "synthetic_constant",
            "n_items_in_batch": len(batch_items),
            "note": (
                "输入是按 bundle schema 生成的合成常量特征（数值列 0.0、类别列固定占位串），"
                "不读取任何真实 CSV 或 test 数据；因此本报告**不是**模型效果证据。"
            ),
        },
        "steps": steps,
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
        },
        "limitations": [
            "模型分数是未校准的模型分数，不是欺诈概率；flagged 只是 score >= 冻结阈值。",
            "smoke 只证明接口链路可用，不构成任何模型效果或生产适用性声明。",
            "服务无认证、限流与 TLS，默认仅监听本机回环地址，不得直接暴露到公网。",
            "本工作包不含数据库、结果查询、持久化幂等、规则引擎、Docker 与 CI。",
        ],
    }

    output_path = resolve_output(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for step in steps:
        print(
            f"  {step['name']:<12} {step['method']:<5} {step['route']:<20} "
            f"-> {step['status_code']}  {step['duration_ms']} ms"
        )
    print(f"smoke 全通过：{report['all_steps_ok']} | 总耗时 {total_ms} ms")
    print(f"报告：{config.relative(output_path)}")
    print("提醒：smoke 分数只是接口连通性证据，不是模型效果指标。")
    return 0 if report["all_steps_ok"] else 1


def _step_extra(name: str, body: Any, runtime: Any) -> dict[str, Any]:
    """把"结论性字段"提出来，但不写入任何载荷或特征取值。"""
    if not isinstance(body, dict):
        return {}
    if name == "model":
        return {
            "n_required_features": len(body.get("required_features", [])),
            "threshold": body.get("threshold"),
            "score_semantics": body.get("score_semantics"),
        }
    if name == "score":
        return {
            "score_finite": isinstance(body.get("score"), (int, float)),
            "score_in_unit_interval": _in_unit_interval(body.get("score")),
            "threshold": body.get("threshold"),
            "flagged": body.get("flagged"),
        }
    if name == "score_batch":
        results = body.get("results", [])
        ids = [item.get("client_item_id") for item in results]
        return {
            "batch_size": body.get("batch_size"),
            "order_preserved": ids == ["smoke-1", "smoke-2"],
            "all_scores_finite": all(
                isinstance(item.get("score"), (int, float)) for item in results
            ),
            "threshold": body.get("threshold"),
        }
    return {}


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
