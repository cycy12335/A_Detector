"""CW-013 离线 bundle 导入入口。

用法（项目根目录下执行）::

    python scripts/import_bundle.py

校验交付包的结构、清单、实际字节大小与摘要，并与**本地** ``configs/quality_gate.yaml``
冻结的摘要比对（清单不能绕过它）；通过后原子写入门禁配置要求的项目相对目标路径。
目标已存在时：摘要相同 → 幂等成功；不同 → **拒绝**且保持原文件字节不变。

退出码（稳定）：0=成功（新建或已存在且摘要相同）；2=字节与冻结摘要不符；
3=交付包结构或清单不合格；4=路径不安全；5=目标已存在且字节不同；6=写入或报告失败。
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
    import_bundle,
    report_path,
)
from project_gate.gate import write_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="import_bundle.py",
        description="把离线交付包导入到门禁配置冻结的目标路径。",
    )
    parser.add_argument("--package-dir", default=PACKAGE_DIR_REL, help="交付包的项目相对目录")
    args = parser.parse_args(argv)

    try:
        outcome = import_bundle(package_dir=args.package_dir)
    except DeliveryError as error:
        outcome = {**error.result, "status": "failed", "error_code": error.code}
        print(f"import      failed  error={error.code}")
        _write(outcome)
        return error.exit_code
    print(
        f"import      {outcome['status']}  target={outcome['target']['path']}"
        f"  size={outcome['target']['size']} B"
    )
    _write(outcome)
    return 0


def _write(outcome: dict) -> None:
    try:
        write_report(outcome, report_path(PROJECT_ROOT, "import_report.json"))
        print("报告：artifacts/reports/cw013_delivery/import_report.json")
    except OSError:
        print("报告写入失败（report_write_failed）")
        raise SystemExit(EXIT_WRITE) from None


if __name__ == "__main__":
    raise SystemExit(main())
