"""CW-022: reproduce four synthetic safety-boundary bad cases without raw payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

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

INPUTS_PATH = PROJECT_ROOT / "configs/cw022_inputs.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/reports/cw022"

IMPROVEMENT_PRIORITIES = [
    {
        "priority": "P1-maintain",
        "item": "持续维护来源 SHA、policy version/digest 绑定",
        "status": "implemented_guard",
        "rationale": "资产漂移会使全部后续证据不可用，必须优先保持失败关闭。",
        "validation": "BC-001/BC-003 与 CW-015/CW-017 来源测试",
    },
    {
        "priority": "P1-maintain",
        "item": "持续维护候选 schema、事实和引用独立校验",
        "status": "implemented_guard",
        "rationale": "不可信候选不得进入 verified 草稿，直接影响证据可信度。",
        "validation": "BC-002 与 CW-017 候选拒绝测试",
    },
    {
        "priority": "P1-maintain",
        "item": "持续维护外部调用预算、重试和超时降级",
        "status": "implemented_guard",
        "rationale": "避免外部故障扩大为无限重试、错误结论或费用失控。",
        "validation": "BC-004 与 CW-017 timeout 测试",
    },
    {
        "priority": "P2-proposed",
        "item": "增加脱敏边界事件计数和运营告警",
        "status": "proposed",
        "rationale": "当前能安全拒绝/降级，但缺少跨案例趋势可观测性；不改变服务协议或风险政策。",
        "validation": "后续新增稳定事件计数和脱敏告警测试",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_inputs() -> dict[str, Any]:
    config = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    if config.get("schema_version") != "cw022_inputs_v1":
        raise ValueError("cw022_inputs_schema_mismatch")
    for relative, expected in config["frozen_sources"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source_drift:{relative}")
    return config


def source_refs(config: dict[str, Any], case_id: str) -> list[dict[str, str]]:
    return [
        {"path": relative, "sha256": config["frozen_sources"][relative]}
        for relative in config["case_sources"][case_id]
    ]


def case_card(
    config: dict[str, Any],
    case_id: str,
    boundary: str,
    input_summary: str,
    expected_status: str,
    observed_status: str,
    error_codes: tuple[str, ...],
    expected_error_codes: tuple[str, ...],
    attempts: int,
    expected_attempts: int,
    calls: int,
    expected_calls: int,
    draft_present: bool,
    expected_draft_present: bool,
    assertion: str,
    root_cause_class: str,
    root_cause_evidence: str,
    impact_boundary: str,
    current_defense: str,
    improvement_status: str,
    reproduction: str,
) -> dict[str, Any]:
    enforce_case_contract(
        case_id,
        expected_status=expected_status,
        observed_status=observed_status,
        error_codes=error_codes,
        expected_error_codes=expected_error_codes,
        attempts=attempts,
        expected_attempts=expected_attempts,
        calls=calls,
        expected_calls=expected_calls,
        draft_present=draft_present,
        expected_draft_present=expected_draft_present,
    )
    return {
        "case_id": case_id,
        "boundary": boundary,
        "input_summary": input_summary,
        "reproduction_command": reproduction,
        "expected_status": expected_status,
        "observed_status": observed_status,
        "expected_error_codes": list(expected_error_codes),
        "observed_error_codes": list(error_codes),
        "expected_attempts": expected_attempts,
        "attempts": attempts,
        "expected_transport_calls": expected_calls,
        "transport_calls": calls,
        "expected_draft_present": expected_draft_present,
        "observed_draft_present": draft_present,
        "assertion": assertion,
        "root_cause_class": root_cause_class,
        "root_cause_evidence": root_cause_evidence,
        "impact_boundary": impact_boundary,
        "current_defense": current_defense,
        "improvement_status": improvement_status,
        "source_refs": source_refs(config, case_id),
    }


def enforce_case_contract(
    case_id: str,
    *,
    expected_status: str,
    observed_status: str,
    error_codes: tuple[str, ...],
    expected_error_codes: tuple[str, ...],
    attempts: int,
    expected_attempts: int,
    calls: int,
    expected_calls: int,
    draft_present: bool,
    expected_draft_present: bool,
) -> None:
    """Make every claim in a case card an executable assertion."""
    if observed_status != expected_status:
        raise AssertionError(f"{case_id}:status_mismatch")
    if error_codes != expected_error_codes:
        raise AssertionError(f"{case_id}:error_codes_mismatch")
    if attempts != expected_attempts:
        raise AssertionError(f"{case_id}:attempts_mismatch")
    if calls != expected_calls:
        raise AssertionError(f"{case_id}:transport_calls_mismatch")
    if draft_present != expected_draft_present:
        raise AssertionError(f"{case_id}:draft_presence_mismatch")


def reproduce_bc001(config: dict[str, Any]) -> dict[str, Any]:
    fixture = json.loads((PROJECT_ROOT / "tests/fixtures/cw015/retrieval_queries.json").read_text(encoding="utf-8"))
    query = next(item for item in fixture["queries"] if item["query_id"] == "q-unsupported")
    hits = build_index().retrieve(query["query"], top_k=5)
    return case_card(
        config,
        "BC-001",
        "retrieval/no_matching_evidence",
        "CW-015 固定合成查询 q-unsupported；查询文本不写入交付结果。",
        "empty",
        "empty" if not hits else "non_empty",
        (),
        (),
        0,
        0,
        0,
        0,
        False,
        False,
        "returned_hit_count == 0",
        "intentional_fail_closed",
        "RetrievalIndex.retrieve 在没有词项命中时返回空元组；CW-015 测试直接断言该行为。",
        "不能生成有来源支撑的候选；不把无证据查询扩写成事实。",
        "固定审查来源清单、citation 校验和空结果分支。",
        "implemented_guard",
        ".venv\\Scripts\\python.exe -X utf8 scripts/run_cw022_bad_cases.py --case BC-001",
    )


def make_snapshot_and_context():
    index = build_index()
    policy = next(item for item in index.chunks if item.authority == "policy")
    snapshot = synthetic_decision("complete_low", policy_sha256=policy.source_sha256, policy_version=policy.source_version)
    return index, snapshot, build_context(snapshot, index)


def reproduce_bc002(config: dict[str, Any]) -> dict[str, Any]:
    index, snapshot, context = make_snapshot_and_context()
    candidate = mutate_candidate(generate_candidate(snapshot, context), "flagged_conflict")
    outcome = ControlledGenerator(GenerationConfig(), OfflineFakeTransport(candidate)).generate(snapshot, context, index)
    return case_card(
        config,
        "BC-002",
        "candidate_validation/contradictory_fact",
        "离线合成候选把 flagged 事实改成与 DecisionSnapshot 相反；不写入候选载荷。",
        "rejected",
        outcome.status,
        outcome.error_codes,
        ("fact_mismatch",),
        outcome.attempts,
        1,
        outcome.calls,
        1,
        outcome.draft is not None,
        False,
        "status == rejected and error_codes == (fact_mismatch,)",
        "intentional_fail_closed",
        "CW-016 validate_candidate 重新从 DecisionSnapshot 校验跨字段事实，CW-017 断言矛盾候选不产生 draft。",
        "候选被拒绝，不进入 verified 草稿；这不是生产模型误报/漏报证据。",
        "候选 schema、事实重算、引用集合和 policy 版本独立校验。",
        "implemented_guard",
        ".venv\\Scripts\\python.exe -X utf8 scripts/run_cw022_bad_cases.py --case BC-002",
    )


def reproduce_bc003(config: dict[str, Any]) -> dict[str, Any]:
    index = build_index()
    policy = next(item for item in index.chunks if item.authority == "policy")
    snapshot = synthetic_decision("complete_low", policy_sha256="0" * 64, policy_version=policy.source_version)
    transport = OfflineFakeTransport({})
    outcome = ControlledGenerator(GenerationConfig(), transport).generate(snapshot, index=index)
    return case_card(
        config,
        "BC-003",
        "policy_or_bundle_drift/policy_digest",
        "合成决策携带错误 policy SHA；不读取或输出原始决策载荷。",
        "degraded",
        outcome.status,
        outcome.error_codes,
        ("policy_digest_mismatch",),
        outcome.attempts,
        0,
        outcome.calls,
        0,
        outcome.draft is not None,
        False,
        "status == degraded, error_codes == (policy_digest_mismatch,), transport_calls == 0",
        "intentional_fail_closed",
        "CW-016 _policy_chunks 比较 snapshot SHA 与已审查 policy chunk；CW-017 断言在 transport 前降级。",
        "调查生成停止，不产生候选；这是配置/资产完整性保护，不是业务模型效果问题。",
        "来源 manifest SHA、policy version 和 policy digest 三重绑定。",
        "implemented_guard",
        ".venv\\Scripts\\python.exe -X utf8 scripts/run_cw022_bad_cases.py --case BC-003",
    )


def reproduce_bc004(config: dict[str, Any]) -> dict[str, Any]:
    index, snapshot, context = make_snapshot_and_context()
    candidate = generate_candidate(snapshot, context)
    transport = OfflineFakeTransport(candidate)
    transport.failures = [ProviderTimeout(), ProviderTimeout()]
    outcome = ControlledGenerator(GenerationConfig(max_retries=1, max_calls=2), transport).generate(snapshot, context, index)
    return case_card(
        config,
        "BC-004",
        "external_model/timeout_degradation",
        "离线 fake transport 连续两次超时；不发起真实外部模型调用。",
        "degraded",
        outcome.status,
        outcome.error_codes,
        ("provider_timeout",),
        outcome.attempts,
        2,
        outcome.calls,
        2,
        outcome.draft is not None,
        False,
        "status == degraded, error_codes == (provider_timeout,), attempts == calls == 2",
        "intentional_fail_closed",
        "ControlledGenerator 只在 max_retries/max_calls 范围内重试，连续 ProviderTimeout 后返回稳定错误码。",
        "没有候选草稿；系统保留降级状态，不把超时当作模型结论。",
        "调用次数、重试上限、超时错误分类和无 draft 返回。",
        "implemented_guard",
        ".venv\\Scripts\\python.exe -X utf8 scripts/run_cw022_bad_cases.py --case BC-004",
    )


CASES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "BC-001": reproduce_bc001,
    "BC-002": reproduce_bc002,
    "BC-003": reproduce_bc003,
    "BC-004": reproduce_bc004,
}


def negative_contract_controls() -> list[dict[str, str]]:
    controls = [
        {
            "control_id": "BC-NEG-001",
            "mutation": "replace provider_timeout with unrelated_error",
            "actual_error_codes": ("unrelated_error",),
            "expected_error_codes": ("provider_timeout",),
            "actual_attempts": 2,
            "expected_attempts": 2,
            "actual_calls": 2,
            "expected_calls": 2,
        },
        {
            "control_id": "BC-NEG-002",
            "mutation": "replace two transport calls with one",
            "actual_error_codes": ("provider_timeout",),
            "expected_error_codes": ("provider_timeout",),
            "actual_attempts": 1,
            "expected_attempts": 2,
            "actual_calls": 1,
            "expected_calls": 2,
        },
    ]
    results = []
    for control in controls:
        try:
            enforce_case_contract(
                control["control_id"],
                expected_status="degraded",
                observed_status="degraded",
                error_codes=control["actual_error_codes"],
                expected_error_codes=control["expected_error_codes"],
                attempts=control["actual_attempts"],
                expected_attempts=control["expected_attempts"],
                calls=control["actual_calls"],
                expected_calls=control["expected_calls"],
                draft_present=False,
                expected_draft_present=False,
            )
        except AssertionError as error:
            results.append(
                {
                    "control_id": control["control_id"],
                    "mutation": control["mutation"],
                    "observed": "contract_rejected",
                    "failure_code": str(error),
                }
            )
        else:
            raise AssertionError(f"{control['control_id']}:negative_control_accepted")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce CW-022 synthetic safety-boundary cases.")
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    args = parser.parse_args()
    try:
        config = load_inputs()
        selected = list(CASES) if args.case == "all" else [args.case]
        cases = [CASES[case_id](config) for case_id in selected]
        negative_controls = negative_contract_controls() if args.case == "all" else []
        result = {
            "schema_version": "cw022_results_v1",
            "run_status": "passed",
            "upstream": config["upstream"],
            "scope": {
                "synthetic_only": True,
                "raw_transactions_read": False,
                "raw_provider_payloads_written": False,
                "deepseek_calls": False,
                "new_training": False,
                "new_test_evaluation": False,
            },
            "cases": cases,
            "negative_contract_controls": negative_controls,
            "improvement_priorities": IMPROVEMENT_PRIORITIES,
            "limitations": [
                "四例均为合成/离线安全边界复现，不是生产事故记录。",
                "CW-009 合成回归语料没有真实业务标签，不能据此声称欺诈误报或漏报。",
                "本包记录主动失败关闭与降级行为，不评价模型业务效果。",
            ],
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "cw022_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "cw022_sources_v1",
            "inputs_config": rel(INPUTS_PATH),
            "frozen_sources": [{"path": path, "sha256": sha} for path, sha in config["frozen_sources"].items()],
            "case_sources": config["case_sources"],
        }
        (OUTPUT_DIR / "cw022_sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run_status": "passed", "case_count": len(cases), "output_dir": rel(OUTPUT_DIR)}, ensure_ascii=False))
        return 0
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"run_status": "failed", "error": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
