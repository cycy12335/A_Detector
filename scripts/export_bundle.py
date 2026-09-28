"""CW-013 离线 bundle 导出入口。

用法（项目根目录下执行）::

    python scripts/export_bundle.py

先把真实 bundle 的流式 SHA-256 与 ``configs/quality_gate.yaml`` 冻结的摘要**按字节**核对，
不符就**不导出**；通过后写出交付包（载荷 + 清单）到 ``artifacts/delivery/``
（被 Git 忽略）。清单只是自描述，不是授权来源。

退出码（稳定）：0=成功（新建或已存在且摘要相同）；2=源缺失或与冻结摘要不符；
3=交付包不合格；4=路径不安全；5=已存在且字节不同；6=写入或报告失败。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_gate.bundle_delivery import (  # noqa: E402
    EXIT_WRITE,
    PACKAGE_DIR_REL,
    DeliveryError,
    export_bundle,
    report_path,
)
from project_gate.gate import write_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_bundle.py",
        description="导出已验收 bundle 为离线交付包（只写被 Git 忽略的目录）。",
    )
    parser.add_argument("--package-dir", default=PACKAGE_DIR_REL, help="交付包的项目相对目录")
    args = parser.parse_args(argv)

    outcome: dict
    try:
        outcome = export_bundle(package_dir=args.package_dir)
    except DeliveryError as error:
        outcome = {**error.result, "status": "failed", "error_code": error.code}
        print(f"export      failed  error={error.code}")
        _write(outcome)
        return error.exit_code
    print(f"export      {outcome['status']}  payload={outcome['payload']['size']} B")
    _write(outcome)
    return 0


def _write(outcome: dict) -> None:
    try:
        write_report(outcome, report_path(PROJECT_ROOT, "export_report.json"))
        print("报告：artifacts/reports/cw013_delivery/export_report.json")
    except OSError:
        print("报告写入失败（report_write_failed）")
        raise SystemExit(EXIT_WRITE) from None


if __name__ == "__main__":
    raise SystemExit(main())
