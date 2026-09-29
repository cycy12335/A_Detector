"""用人工审查的合成查询集计算 CW-015 检索指标。"""

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

from ai_detector.retrieval import RetrievalError, SourceManifestError, build_index  # noqa: E402

QUERIES_PATH = PROJECT_ROOT / "tests/fixtures/cw015/retrieval_queries.json"


def _load_queries() -> dict[str, Any]:
    raw = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "cw015_queries_v1" or not isinstance(raw.get("queries"), list):
        raise ValueError("查询集 schema 不合法")
    return raw


def evaluate(*, top_k: int = 5) -> dict[str, Any]:
    index = build_index()
    dataset = _load_queries()
    queries = dataset["queries"]
    recall_at_1_total = 0.0
    recall_at_k_total = 0.0
    reciprocal_rank = 0.0
    positive_total = 0
    negative_total = 0
    negative_empty = 0
    details: list[dict[str, Any]] = []
    for item in queries:
        expected = set(item["relevant_ids"])
        hits = index.retrieve(item["query"], top_k=top_k)
        returned = [hit.chunk.chunk_id for hit in hits]
        ranks = [position for position, citation_id in enumerate(returned, start=1) if citation_id in expected]
        if expected:
            positive_total += 1
            recall_at_1_total += int(bool(returned) and returned[0] in expected) / len(expected)
            recall_at_k_total += len(expected.intersection(returned)) / len(expected)
            if ranks:
                reciprocal_rank += 1.0 / ranks[0]
        if item["expected_empty"]:
            negative_total += 1
            negative_empty += int(not returned)
        details.append({"query_id": item["query_id"], "returned_ids": returned, "first_relevant_rank": ranks[0] if ranks else None})
    count = len(queries)
    return {
        "schema_version": "cw015_evaluation_v1",
        "dataset_version": dataset["dataset_version"],
        "knowledge_base_policy_versions": list(index.policy_versions),
        "n_queries": count,
        "positive_queries": positive_total,
        "top_k": top_k,
        "recall_at_1": recall_at_1_total / positive_total if positive_total else 0.0,
        "recall_at_k": recall_at_k_total / positive_total if positive_total else 0.0,
        "mrr": reciprocal_rank / positive_total if positive_total else 0.0,
        "negative_queries": negative_total,
        "negative_empty_rate": negative_empty / negative_total if negative_total else 0.0,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CW-015 offline retrieval on reviewed synthetic queries.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(top_k=args.top_k)
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0
    except (SourceManifestError, RetrievalError, ValueError, OSError) as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
