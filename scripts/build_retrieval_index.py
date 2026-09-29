"""构建并验证 CW-015 的离线证据索引。"""

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
    parser = argparse.ArgumentParser(description="Build the verified CW-015 offline evidence index.")
    parser.add_argument("--output", type=Path, help="optional project-relative JSON output path")
    args = parser.parse_args()
    try:
        index = build_index()
        payload = index.as_dict()
        if args.output:
            output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", "chunks": len(index.chunks), "policy_versions": list(index.policy_versions)}, ensure_ascii=False))
        return 0
    except (SourceManifestError, RetrievalError) as error:
        print(json.dumps({"status": "failed", "error": error.code}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
