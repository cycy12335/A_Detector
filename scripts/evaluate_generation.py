"""CW-017 离线合成评测；不调用网络、不读取凭据、不保存请求或响应。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.investigation import (  # noqa: E402
    ControlledGenerator,
    GenerationConfig,
    OfflineFakeTransport,
    ProviderTimeout,
    build_context,
    generate_candidate,
    mutate_candidate,
    synthetic_decision,
)
from ai_detector.retrieval import build_index  # noqa: E402

CASES_PATH = PROJECT_ROOT / "tests/fixtures/cw017/generation_cases.json"


class _MalformedTransport:
    def complete(self, request: Mapping[str, Any], *, timeout_seconds: float) -> str:
        return '{"schema_version":"wrong","candidate":{}}'


def _cases() -> dict[str, Any]:
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "cw017_cases_v1" or not isinstance(raw.get("cases"), list):
        raise ValueError("CW-017 cases schema invalid")
    return raw


def evaluate() -> dict[str, Any]:
    dataset = _cases()
    index = build_index()
    policy = next(chunk for chunk in index.chunks if chunk.authority == "policy")
    outcomes: list[tuple[str, str, bool, tuple[str, ...]]] = []
    for case in dataset["cases"]:
        snapshot = synthetic_decision(case["decision_variant"], policy_sha256=policy.source_sha256, policy_version=policy.source_version)
        context = build_context(snapshot, index)
        candidate = generate_candidate(snapshot, context)
        kind = case["kind"]
        config = GenerationConfig()
        if case["mutation"]:
            candidate = mutate_candidate(candidate, case["mutation"])
        if kind == "timeout":
            transport = OfflineFakeTransport(candidate, failures=[ProviderTimeout(), ProviderTimeout()])
            config = GenerationConfig(max_retries=1, max_calls=2)
        elif kind == "missing_credentials":
            transport = OfflineFakeTransport(candidate)
            config = GenerationConfig(mode="real", provider_name="selected-provider", model_name="selected-model", credential_env="CW017_MISSING_KEY", max_cost_usd=1.0)
        elif kind == "malformed_response":
            transport = _MalformedTransport()
        elif kind == "cost_limit":
            transport = OfflineFakeTransport(candidate, cost_usd=0.01)
        else:
            transport = OfflineFakeTransport(candidate)
        outcome = ControlledGenerator(config, transport).generate(snapshot, context, index)
        expected = case["expected_status"]
        correct = outcome.status == expected
        outcomes.append((outcome.status, expected, correct, tuple(case["checks"])))
    count = len(outcomes)
    fact_cases = sum("fact" in checks for _actual, _expected, _correct, checks in outcomes)
    evidence_cases = sum("evidence" in checks for _actual, _expected, _correct, checks in outcomes)
    format_correct = sum(correct for _actual, _expected, correct, _checks in outcomes)
    fact_correct = sum(correct for _actual, _expected, correct, checks in outcomes if "fact" in checks)
    evidence_correct = sum(correct for _actual, _expected, correct, checks in outcomes if "evidence" in checks)
    return {
        "schema_version": "cw017_evaluation_v1",
        "dataset_version": dataset["dataset_version"],
        "offline_only": True,
        "real_e2e_evidence": False,
        "n_cases": count,
        "expected_status_counts": {status: sum(expected == status for _actual, expected, _correct, _checks in outcomes) for status in ("verified", "insufficient_evidence", "rejected", "degraded")},
        "observed_status_counts": {status: sum(actual == status for actual, _expected, _correct, _checks in outcomes) for status in ("verified", "insufficient_evidence", "rejected", "degraded")},
        "metric_semantics": "expected_status_classification_on_offline_fault_cases",
        "format_cases": count,
        "format_success_count": format_correct,
        "format_success_rate": format_correct / count if count else 0.0,
        "fact_cases": fact_cases,
        "fact_consistency_count": fact_correct,
        "fact_consistency_rate": fact_correct / fact_cases if fact_cases else 0.0,
        "evidence_cases": evidence_cases,
        "evidence_completeness_count": evidence_correct,
        "evidence_completeness_rate": evidence_correct / evidence_cases if evidence_cases else 0.0,
        "policy_version": policy.source_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the offline CW-017 controlled generation boundary.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate()
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0
    except (OSError, ValueError, KeyError, StopIteration):
        print(json.dumps({"status": "failed", "error": "evaluation_failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
