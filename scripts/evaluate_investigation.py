"""CW-016 合成结构化调查草稿评测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.investigation import build_context, generate_candidate, mutate_candidate, synthetic_decision, validate_candidate  # noqa: E402
from ai_detector.retrieval import build_index  # noqa: E402

CASES_PATH = PROJECT_ROOT / "tests/fixtures/cw016/investigation_cases.json"


def _load_cases() -> dict[str, Any]:
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "cw016_cases_v1" or not isinstance(raw.get("cases"), list):
        raise ValueError("CW-016 cases schema invalid")
    return raw


def evaluate() -> dict[str, Any]:
    dataset = _load_cases()
    index = build_index()
    policy_chunk = next(chunk for chunk in index.chunks if chunk.authority == "policy")
    outcomes: list[tuple[str, str, bool, tuple[str, ...]]] = []
    for case in dataset["cases"]:
        snapshot = synthetic_decision(case["decision_variant"], policy_sha256=policy_chunk.source_sha256, policy_version=policy_chunk.source_version)
        context = build_context(snapshot, index)
        candidate = generate_candidate(snapshot, context)
        if case["candidate"] != "generated":
            candidate = mutate_candidate(candidate, case["candidate"])
        result = validate_candidate(candidate, snapshot, context, index)
        expected = case["expected_status"]
        expected_error = case["expected_error_code"]
        correct = (
            result.status == expected
            and result.error_codes == ((expected_error,) if expected_error else ())
            and (result.draft is not None) == (expected != "rejected")
        )
        outcomes.append((result.status, expected, correct, tuple(case["checks"])))
    count = len(outcomes)
    fact_total = sum("fact" in checks for _actual, _expected, _correct, checks in outcomes)
    evidence_total = sum("evidence" in checks for _actual, _expected, _correct, checks in outcomes)
    return {
        "schema_version": "cw016_evaluation_v1",
        "dataset_version": dataset["dataset_version"],
        "n_cases": count,
        "expected_status_counts": {status: sum(expected == status for _actual, expected, _correct, _checks in outcomes) for status in ("verified", "insufficient_evidence", "rejected")},
        "observed_status_counts": {status: sum(actual == status for actual, _expected, _correct, _checks in outcomes) for status in ("verified", "insufficient_evidence", "rejected")},
        "format_cases": count,
        "fact_cases": fact_total,
        "evidence_cases": evidence_total,
        "format_success_rate": sum(correct for _actual, _expected, correct, _checks in outcomes) / count if count else 0.0,
        "fact_consistency_rate": sum(correct for _actual, _expected, correct, checks in outcomes if "fact" in checks) / fact_total if fact_total else 0.0,
        "evidence_completeness_rate": sum(correct for _actual, _expected, correct, checks in outcomes if "evidence" in checks) / evidence_total if evidence_total else 0.0,
        "policy_version": policy_chunk.source_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CW-016 structured investigation draft validation.")
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
