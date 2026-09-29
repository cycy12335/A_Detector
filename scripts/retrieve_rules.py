"""CW-015 离线检索 CLI；输出仅包含已验证来源的片段和引用。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.retrieval import RetrievalError, SourceManifestError, build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve verified CW-015 rule evidence offline.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-output-chars", type=int, default=4_000)
    args = parser.parse_args()
    try:
        hits = build_index().retrieve(args.query, top_k=args.top_k, max_output_chars=args.max_output_chars)
        print(json.dumps({"query": args.query, "evidence": [hit.as_dict() for hit in hits]}, ensure_ascii=False, indent=2))
        return 0
    except (SourceManifestError, RetrievalError) as error:
        print(json.dumps({"status": "failed", "error": error.code}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
