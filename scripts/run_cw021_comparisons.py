"""CW-021: aggregate historical model/cost evidence and run fixed CW-015 top-k comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.retrieval import build_index  # noqa: E402

INPUTS_PATH = PROJECT_ROOT / "configs/cw021_inputs.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/reports/cw021"
MODEL_FIELDS = (
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "roc_auc",
    "pr_auc",
    "predicted_positive_rate",
    "frozen_threshold",
    "train_seconds",
)


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
    if config.get("schema_version") != "cw021_inputs_v1":
        raise ValueError("CW-021 输入清单 schema 不合法")
    for relative, expected in config["frozen_sources"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source_drift:{relative}")
    return config


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float:
    return float(value)


def historical_costs(report: dict[str, Any], model_key: str) -> dict[str, Any]:
    if "baselines" in report:
        source = report["baselines"][model_key]["test"]["cost_scenarios"]
    else:
        source = report["test_evaluation"]["cost_scenarios"]
    result: dict[str, Any] = {}
    for scenario, payload in source.items():
        result[scenario] = {
            "fp_cost": payload["fp_cost"],
            "fn_cost": payload["fn_cost"],
            "frozen_threshold": payload["frozen_threshold"],
            "test_total_cost": payload["test_total_cost"],
            "test_average_cost": payload["test_average_cost"],
        }
    return result


def model_comparison(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "artifacts/reports/cw004_lightgbm/comparison.csv"
    rows = read_csv(path)
    expected = {"rule_tail_anomaly", "dummy", "logistic_regression", "lightgbm"}
    if {row["model_key"] for row in rows} != expected:
        raise ValueError("model_comparison_model_set_mismatch")
    output = []
    for row in rows:
        package = row["source_work_package"]
        report_path = PROJECT_ROOT / (
            "artifacts/reports/cw003_ieee_cis_baselines/report.json"
            if package == "CW-003"
            else "artifacts/reports/cw004_lightgbm/report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        values: dict[str, Any] = {"model_key": row["model_key"]}
        values.update({field: number(row[field]) for field in MODEL_FIELDS})
        values.update(
            {
                "source_work_package": package,
                "metrics_source": row["metrics_source"],
                "source_report_sha256_declared": row["source_report_sha256"],
                "source_report_file_sha256": sha256_file(report_path),
                "complexity_note": row["complexity_note"],
                "cost_basis": "historical_test_after_validation_freeze",
                "cost_unit": "research_relative_cost_not_currency",
                "cost_scenarios": historical_costs(report, row["model_key"]),
            }
        )
        output.append(values)
    return output


def curve_rows(path: Path) -> dict[str, list[dict[str, float]]]:
    rows = read_csv(path)
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        if row["partition"] != "validation":
            raise ValueError(f"threshold_curve_non_validation_row:{rel(path)}")
        grouped.setdefault(row["model_key"], []).append(
            {
                "threshold": number(row["threshold"]),
                "true_positive": number(row["true_positive"]),
                "false_positive": number(row["false_positive"]),
                "false_negative": number(row["false_negative"]),
                "false_positive_rate": number(row["false_positive_rate"]),
                "recall": number(row["recall"]),
                "f1": number(row["f1"]),
                "n_samples": number(row["n_samples"]),
            }
        )
    return grouped


def threshold_cost_comparison(config: dict[str, Any]) -> list[dict[str, Any]]:
    curves = {}
    for relative in (
        "artifacts/reports/cw003_ieee_cis_baselines/threshold_curve.csv",
        "artifacts/reports/cw004_lightgbm/threshold_curve.csv",
    ):
        curves.update(curve_rows(PROJECT_ROOT / relative))
    output = []
    for model_key, rows in sorted(curves.items()):
        f1_row = max(rows, key=lambda item: (item["f1"], item["recall"], item["threshold"]))
        for scenario, costs in config["cost_scenarios"].items():
            scored = []
            for row in rows:
                total = row["false_positive"] * costs["fp_cost"] + row["false_negative"] * costs["fn_cost"]
                scored.append((total, -row["threshold"], row))
            total, _, selected = min(scored, key=lambda item: (item[0], item[1]))
            output.append(
                {
                    "model_key": model_key,
                    "scenario": scenario,
                    "selection_partition": "validation",
                    "selection_rule": "minimum validation total cost; ties choose higher threshold",
                    "fp_cost": costs["fp_cost"],
                    "fn_cost": costs["fn_cost"],
                    "selected_threshold": selected["threshold"],
                    "validation_recall": selected["recall"],
                    "validation_false_positive_rate": selected["false_positive_rate"],
                    "validation_false_positive": selected["false_positive"],
                    "validation_false_negative": selected["false_negative"],
                    "validation_total_cost": total,
                    "validation_average_cost": total / selected["n_samples"],
                    "validation_f1_threshold": f1_row["threshold"],
                    "validation_f1_at_f1_threshold": f1_row["f1"],
                    "cost_unit": "research_relative_cost_not_currency",
                }
            )
    return output


def retrieval_comparison(config: dict[str, Any]) -> list[dict[str, Any]]:
    fixture = json.loads((PROJECT_ROOT / "tests/fixtures/cw015/retrieval_queries.json").read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "cw015_queries_v1":
        raise ValueError("retrieval_fixture_schema_mismatch")
    index = build_index()
    output = []
    for top_k in config["retrieval_top_k"]:
        recall_at_1 = 0.0
        recall_at_k = 0.0
        mrr = 0.0
        positive = 0
        negative = 0
        negative_empty = 0
        evidence_chars = 0
        citation_count = 0
        positive_evidence_chars = 0
        positive_citation_count = 0
        for item in fixture["queries"]:
            expected = set(item["relevant_ids"])
            hits = index.retrieve(item["query"], top_k=top_k)
            returned = [hit.chunk.chunk_id for hit in hits]
            evidence_chars_for_query = sum(len(hit.chunk.text) for hit in hits)
            citations_for_query = len(set(returned))
            evidence_chars += evidence_chars_for_query
            citation_count += citations_for_query
            if expected:
                positive += 1
                positive_evidence_chars += evidence_chars_for_query
                positive_citation_count += citations_for_query
                ranks = [rank for rank, item_id in enumerate(returned, start=1) if item_id in expected]
                recall_at_1 += int(bool(returned) and returned[0] in expected) / len(expected)
                recall_at_k += len(expected.intersection(returned)) / len(expected)
                if ranks:
                    mrr += 1.0 / ranks[0]
            if item["expected_empty"]:
                negative += 1
                negative_empty += int(not returned)
        count = len(fixture["queries"])
        output.append(
            {
                "top_k": top_k,
                "dataset_version": fixture["dataset_version"],
                "n_queries": count,
                "positive_queries": positive,
                "negative_queries": negative,
                "recall_at_1": recall_at_1 / positive,
                "recall_at_k": recall_at_k / positive,
                "mrr": mrr / positive,
                "negative_empty_rate": negative_empty / negative,
                "avg_evidence_chars_all_queries": evidence_chars / count,
                "avg_citation_count_all_queries": citation_count / count,
                "avg_evidence_chars_positive_queries": positive_evidence_chars / positive,
                "avg_citation_count_positive_queries": positive_citation_count / positive,
                "evidence_measure": "sum of returned verified chunk text lengths; aggregate only",
                "citation_measure": "unique returned citation IDs; aggregate only",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "cost_scenarios"}
        for scenario, payload in row["cost_scenarios"].items():
            for key, value in payload.items():
                item[f"{scenario}_{key}"] = value
        flattened.append(item)
    return flattened


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible CW-021 comparison aggregations.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        config = load_inputs()
        model_rows = model_comparison(config)
        threshold_rows = threshold_cost_comparison(config)
        retrieval_rows = retrieval_comparison(config)
        output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        source_manifest = {
            "schema_version": "cw021_sources_v1",
            "inputs_config": rel(INPUTS_PATH),
            "frozen_sources": [
                {"path": key, "sha256": value} for key, value in config["frozen_sources"].items()
            ],
        }
        results = {
            "schema_version": "cw021_results_v1",
            "status": "passed",
            "command": "python scripts/run_cw021_comparisons.py",
            "scope": {
                "raw_transactions_read": False,
                "new_training": False,
                "new_test_evaluation": False,
                "deepseek_calls": False,
            },
            "model_comparison": model_rows,
            "threshold_cost_comparison": threshold_rows,
            "retrieval_top_k_comparison": retrieval_rows,
            "sources": source_manifest,
            "limitations": [
                "模型组是 CW-003/CW-004 跨来源历史观察性对比，不是同轮重训或单因素消融。",
                "阈值组只使用 validation 曲线，不依据 test 选点。",
                "检索组只改变 top-k，使用固定合成查询和同一审查来源快照。",
                "模型分数不是欺诈概率，成本是研究相对单位，不代表货币损失或生产收益。",
            ],
        }
        (output_dir / "cw021_sources.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "cw021_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_csv(output_dir / "cw021_model_comparison.csv", flatten_model_rows(model_rows))
        write_csv(output_dir / "cw021_threshold_cost.csv", threshold_rows)
        write_csv(output_dir / "cw021_retrieval_topk.csv", retrieval_rows)
        print(json.dumps({"status": "passed", "output_dir": rel(output_dir), "retrieval_rows": len(retrieval_rows)}, ensure_ascii=False))
        return 0
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"status": "failed", "error": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
