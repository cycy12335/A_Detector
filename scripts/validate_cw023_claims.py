"""CW-023 简历 claim、冻结来源和复现口径的失败即停验证器。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS_PATH = PROJECT_ROOT / "configs" / "cw023_inputs.json"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "cw023"
REPORT_PATH = REPORT_DIR / "cw023_claim_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_path(value: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise AssertionError(f"missing json path: {path}")
        value = value[part]
    return value


def equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and abs(float(actual) - expected) <= 1e-12
    return actual == expected


def assert_text_token(document: str, token: str, claim_id: str) -> None:
    if token not in document:
        raise AssertionError(f"{claim_id}: text token missing: {token}")


def validate_assertion(assertion: dict[str, Any], source_path: Path, source: Any, document: str) -> None:
    kind = assertion["kind"]
    expected = assertion.get("expected")
    if kind == "json_equals":
        actual = get_path(source, assertion["path"])
    elif kind == "json_list_count":
        actual = len(get_path(source, assertion["path"]))
    elif kind == "retrieval_row":
        rows = source["retrieval_top_k_comparison"]
        row = next((item for item in rows if item["top_k"] == assertion["top_k"]), None)
        if row is None:
            raise AssertionError(f"missing retrieval top_k={assertion['top_k']}")
        actual = row[assertion["field"]]
    elif kind == "case_status":
        row = next((item for item in source["cases"] if item["case_id"] == assertion["case_id"]), None)
        if row is None:
            raise AssertionError(f"missing case: {assertion['case_id']}")
        actual = row[assertion["field"]]
    elif kind == "text_contains":
        actual = assertion["expected"] in document
        expected = True
    else:
        raise AssertionError(f"unsupported assertion kind: {kind}")
    if not equal(actual, expected):
        raise AssertionError(
            f"{source_path.as_posix()} {kind} expected {expected!r}, got {actual!r}"
        )


def validate(inputs: dict[str, Any], *, mutate_for_test: bool = False) -> dict[str, Any]:
    document_path = PROJECT_ROOT / inputs["claims_document"]
    document = document_path.read_text(encoding="utf-8")
    actual_document_sha = sha256(document_path)
    if actual_document_sha != inputs["claims_document_sha256"]:
        raise AssertionError(
            f"claim document drift: expected {inputs['claims_document_sha256']}, got {actual_document_sha}"
        )
    frozen_results = []
    for frozen in inputs["frozen_sources"]:
        path = PROJECT_ROOT / frozen["path"]
        if not path.is_file():
            raise AssertionError(f"frozen source missing: {frozen['path']}")
        actual = sha256(path)
        if actual != frozen["sha256"]:
            raise AssertionError(
                f"source drift: {frozen['path']} expected {frozen['sha256']}, got {actual}"
            )
        frozen_results.append({"path": frozen["path"], "sha256": actual, "match": True})

    source_cache: dict[str, Any] = {}
    source_paths = {item["path"] for item in inputs["frozen_sources"]}
    claim_results = []
    for claim in inputs["claims"]:
        if claim["source_path"] not in source_paths:
            raise AssertionError(f"{claim['id']}: source not frozen: {claim['source_path']}")
        assert_text_token(document, claim["text_token"], claim["id"])
        source_path = PROJECT_ROOT / claim["source_path"]
        if claim["source_path"] not in source_cache:
            source_cache[claim["source_path"]] = (
                load_json(source_path) if source_path.suffix == ".json" else None
            )
        source = source_cache[claim["source_path"]]
        for assertion in claim["assertions"]:
            if assertion["kind"] == "text_contains":
                source_text = source_path.read_text(encoding="utf-8")
                validate_assertion(assertion, source_path, source, source_text)
            elif source is None:
                raise AssertionError(f"unsupported non-json assertion source: {claim['source_path']}")
            else:
                validate_assertion(assertion, source_path, source, document)
        claim_results.append({"claim_id": claim["id"], "source_path": claim["source_path"], "status": "passed"})

    # A deliberately mutated token is checked in every run so regression can prove fail-closed behavior.
    negative_token = inputs["claims"][0]["text_token"] + "__MUTATED__"
    try:
        assert_text_token(document, negative_token, "CW023-NEG-TEXT")
    except AssertionError:
        negative_status = "rejected"
    else:
        raise AssertionError("negative text mutation was not rejected")

    return {
        "schema_version": "cw023_claim_validation_v1",
        "status": "passed",
        "claim_count": len(claim_results),
        "claims": claim_results,
        "frozen_sources": frozen_results,
        "negative_controls": [{"control_id": "CW023-NEG-TEXT", "status": negative_status}],
        "limitations": [
            "来源漂移、缺失 claim token、错误数值或错误分区都会使本命令非零退出。",
            "CW-009 回归集与 CW-022 边界案例均为合成/离线证据，不能写成真实业务 FP/FN 或生产事故。",
            "CW-004 F1 是 IEEE-CIS 历史 test 分区的一次性研究评估，不是校准概率、生产收益或新一轮训练结果。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=INPUTS_PATH)
    args = parser.parse_args()
    try:
        inputs = load_json(args.inputs)
        report = validate(inputs)
    except (AssertionError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"CW-023 validation failed: {exc}", file=sys.stderr)
        return 1
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "claim_count": report["claim_count"], "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
