"""Verify a running CW-018 Compose demo with synthetic input only.

This script never reads real transaction data or the default SQLite database. It loads only
the accepted bundle metadata needed to build the existing synthetic constant-feature payload.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_demo import preflight  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证运行中的 CW-018 Docker 演示。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--config", type=Path, default=Path("configs/demo_service.yaml"))
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser


def _request(base_url: str, method: str, route: str, payload: Any, timeout: float):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + route,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        return error.code, json.loads(body) if body else {}


def _synthetic_features(runtime: Any) -> dict[str, Any]:
    numeric = set(getattr(runtime, "numeric_features", []) or [])
    return {
        name: 0.0 if name in numeric else "compose_demo_constant"
        for name in runtime.source_features
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, runtime, policy = preflight(args.config)
    features = _synthetic_features(runtime)
    checks: list[dict[str, Any]] = []

    status, body = _request(args.base_url, "GET", "/health/live", None, args.timeout)
    checks.append({"name": "live", "status": status, "ok": status == 200 and body.get("status") == "live"})

    status, body = _request(args.base_url, "GET", "/health/ready", None, args.timeout)
    checks.append({"name": "ready", "status": status, "ok": status == 200 and body.get("status") == "ready"})

    status, body = _request(args.base_url, "GET", "/v1/model", None, args.timeout)
    checks.append(
        {
            "name": "model",
            "status": status,
            "ok": status == 200
            and body.get("bundle_sha256") == runtime.bundle_sha256
            and len(body.get("required_features", [])) == 199,
        }
    )

    status, body = _request(args.base_url, "GET", "/v1/policy", None, args.timeout)
    checks.append(
        {
            "name": "policy",
            "status": status,
            "ok": status == 200
            and body.get("policy_sha256") == config.policy_sha256
            and body.get("policy_version") == policy.policy_version
            and len(body.get("rules", [])) == len(policy.rules),
        }
    )

    status, body = _request(args.base_url, "POST", "/v1/score", {"features": features}, args.timeout)
    checks.append(
        {
            "name": "score",
            "status": status,
            "ok": status == 200
            and isinstance(body.get("score"), (int, float))
            and "probability" not in body,
        }
    )

    status, body = _request(args.base_url, "POST", "/v1/score", {"features": {}}, args.timeout)
    checks.append(
        {
            "name": "invalid_features",
            "status": status,
            "ok": status == 422 and body.get("error", {}).get("code") == "feature_contract_violation",
        }
    )

    return {
        "schema_version": "cw018_demo_verification_v1",
        "checks": checks,
        "all_checks_ok": all(item["ok"] for item in checks),
        "synthetic_input": "constant_features_only",
        "storage_used": False,
        "llm_used": False,
        "bundle_sha256": runtime.bundle_sha256,
        "n_features": len(runtime.source_features),
        "config_name": config.name,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except Exception as error:  # noqa: BLE001 - no payload/path details in CLI output
        print(f"demo_verification_failed:{type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
