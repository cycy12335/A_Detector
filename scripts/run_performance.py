"""运行 CW-011 本机性能测量：python scripts/run_performance.py [--smoke]。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from performance.runner import run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CW-011 本机 Locust 性能测量")
    parser.add_argument("--smoke", action="store_true", help="仅运行短时诊断档位")
    args = parser.parse_args(argv)
    code = run(smoke=args.smoke)
    label = "smoke_report" if args.smoke else "report"
    print(f"CW-011 性能测量退出码 {code}；报告：artifacts/reports/cw011_performance/{label}.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
