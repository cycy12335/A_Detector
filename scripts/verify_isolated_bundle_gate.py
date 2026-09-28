"""CW-013 隔离源码副本端到端验证入口。

用法（项目根目录下执行）::

    # 只做隔离副本 + 解释器 + 解析路径 + "缺 bundle 必须失败" 的反例（快）
    python scripts/verify_isolated_bundle_gate.py

    # 再加上：把离线交付包搬进副本、导入、并真实运行原样总门禁（慢，约 2 分钟）
    python scripts/verify_isolated_bundle_gate.py --full

它做的事实：

1. 在项目内被忽略的暂存区构造**不带 ``artifacts/**``** 的已提交源码副本；
2. 建独立 venv，让 ``ai_detector`` 与 ``PROJECT_ROOT`` 都解析到**副本**（打印实际路径）；
3. 先跑"缺 bundle"反例：预检与整体必须失败，且不得变绿；
4. ``--full`` 时把交付包搬进副本 → 导入 → 运行原样 ``python scripts/run_project_gate.py``，
   记录静态、回归与全量 pytest 的真实计数；
5. 复核**真实工作目录**的冻结文件与本轮无关产物未变。

这不是 hosted CI：没有远端、没有流水线文件，副本也只是已提交源码的副本。
退出码：0=全部符合预期；2=预期不符（解析回指、缺失未失败、门禁未通过等）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_gate.bundle_delivery import (  # noqa: E402
    PACKAGE_DIR_REL,
    export_bundle,
)
from project_gate.isolated_check import (  # noqa: E402
    StageError,
    build_isolated_copy,
    create_isolated_venv,
    probe_resolution,
    provide_extra_ignored_inputs,
    read_report,
    run_in_copy,
    safe_stage_path,
)
from project_gate.gate import write_report  # noqa: E402

#: 暂存根（项目内、被 Git 忽略；必须是 ``artifacts/tmp_tests/`` 之下的专用子目录）。
STAGE_REL = "artifacts/tmp_tests/cw013"

#: 真实工作目录中必须保持不变的冻结文件。
FROZEN_REL = (
    "configs/service.yaml",
    "configs/risk_policy.yaml",
    "configs/quality_gate.yaml",
    "tests/fixtures/cw009/model_regression_cases.json",
    "artifacts/models/cw004/lightgbm_bundle.joblib",
    "artifacts/runtime/cw007_decisions.sqlite3",
)

CW009_REPORT_REL = "artifacts/reports/cw009_quality_gate/report.json"
CW012_REPORT_REL = "artifacts/reports/cw012_project_gate/report.json"

EXIT_OK = 0
EXIT_MISMATCH = 2


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_isolated_bundle_gate.py",
        description="在隔离源码副本里验证缺 bundle 失败与导入后原样总门禁通过。",
    )
    parser.add_argument("--stage", default=STAGE_REL, help="暂存目录（项目相对）")
    parser.add_argument("--full", action="store_true", help="额外运行完整的隔离总门禁")
    args = parser.parse_args(argv)

    before = {relative: _digest(PROJECT_ROOT / relative) for relative in FROZEN_REL}
    report: dict = {"work_package": "CW-013", "mode": "full" if args.full else "quick"}

    # 暂存边界：先校验，再创建**本轮专用**目录。任何递归删除都只发生在这个新目录里，
    # 不会碰项目外位置、已提交目录或既有的非本轮目录。
    try:
        stage = safe_stage_path(args.stage, root=PROJECT_ROOT)
    except StageError as error:
        print(f"stage       rejected  error={error.code}")
        report["stage"] = {"requested": args.stage, "error_code": error.code}
        return _finish(report, EXIT_MISMATCH, "暂存路径不在被忽略的暂存区之下")
    round_dir = stage / f"run-{uuid.uuid4().hex}"
    copy_root = round_dir / "checkout"
    report["stage"] = {
        "root": stage.relative_to(PROJECT_ROOT).as_posix(),
        "round": round_dir.relative_to(PROJECT_ROOT).as_posix(),
    }

    try:
        try:
            build_isolated_copy(PROJECT_ROOT, copy_root)
            report["copy_entries"] = sorted(item.name for item in copy_root.iterdir())
            report["copy_has_artifacts"] = (copy_root / "artifacts").exists()
            return _run(report, copy_root, args, before)
        finally:
            # 只清理本轮专用目录（含其中的隔离 venv 与副本）。
            shutil.rmtree(round_dir, ignore_errors=True)
    except StageError as error:
        print(f"stage       rejected  error={error.code}")
        report["stage"]["error_code"] = error.code
        return _finish(report, EXIT_MISMATCH, "暂存路径不在被忽略的暂存区之下")


def _run(report: dict, copy_root: Path, args, before: dict) -> int:
    """副本已就绪后的验证流程。"""
    interpreter = create_isolated_venv(copy_root)
    resolution = probe_resolution(interpreter, copy_root)
    report["resolution"] = resolution
    if not resolution.get("ok"):
        return _finish(report, EXIT_MISMATCH, "隔离解释器无法导入 ai_detector")
    if not (
        resolution["module_inside_copy"]
        and resolution["root_is_copy"]
        and resolution["outside_real_repo"]
    ):
        return _finish(report, EXIT_MISMATCH, "模块解析没有指向隔离副本")

    # --- 反例：缺 bundle 必须失败 ---
    missing = run_in_copy(copy_root, interpreter, ["scripts/run_project_gate.py"], timeout_s=1800)
    missing_report = read_report(copy_root, CW012_REPORT_REL)
    report["missing_bundle"] = {
        "exit_code": missing.returncode,
        "overall_status": (missing_report or {}).get("status"),
        "error_code": (missing_report or {}).get("error_code"),
        "cw009_error": ((missing_report or {}).get("phases", {}).get("quality_gate") or {}).get(
            "cw009_status"
        ),
        "inputs_match": ((missing_report or {}).get("phases", {}).get("quality_gate") or {}).get(
            "inputs_match"
        ),
    }
    if missing.returncode != 5:
        return _finish(report, EXIT_MISMATCH, "缺 bundle 时总门禁没有失败关闭")
    if (missing_report or {}).get("status") != "failed":
        return _finish(report, EXIT_MISMATCH, "缺 bundle 时报告没有标记失败")
    inputs_match = report["missing_bundle"]["inputs_match"] or {}
    if inputs_match.get("bundle") is not False:
        return _finish(report, EXIT_MISMATCH, "缺 bundle 时 bundle 输入被报告为匹配")
    # 反例产物清掉，避免干扰后续导入后的正式运行
    shutil.rmtree(copy_root / "artifacts", ignore_errors=True)

    if not args.full:
        return _finish(report, EXIT_OK, None)

    # --- 正式流程：导出 → 搬进副本 → 导入 → 原样总门禁 ---
    export = export_bundle()
    package_source = PROJECT_ROOT / PACKAGE_DIR_REL
    package_target = copy_root / PACKAGE_DIR_REL
    package_target.parent.mkdir(parents=True, exist_ok=True)
    if package_target.exists():
        shutil.rmtree(package_target)
    shutil.copytree(package_source, package_target)
    report["export"] = export

    # 全量套件额外需要的被忽略产物（见 isolated_check.EXTRA_IGNORED_INPUTS）
    report["extra_ignored_inputs"] = provide_extra_ignored_inputs(PROJECT_ROOT, copy_root)

    imported = run_in_copy(copy_root, interpreter, ["scripts/import_bundle.py"], timeout_s=600)
    import_report = read_report(copy_root, "artifacts/reports/cw013_delivery/import_report.json")
    report["import"] = {"exit_code": imported.returncode, "report": import_report}
    if imported.returncode != 0:
        return _finish(report, EXIT_MISMATCH, "隔离副本里导入交付包失败")

    gate = run_in_copy(copy_root, interpreter, ["scripts/run_project_gate.py"], timeout_s=3600)
    gate_report = read_report(copy_root, CW012_REPORT_REL)
    phases = (gate_report or {}).get("phases", {})
    report["gate"] = {
        "exit_code": gate.returncode,
        "status": (gate_report or {}).get("status"),
        "static": phases.get("static"),
        "quality_gate": {
            "status": (phases.get("quality_gate") or {}).get("status"),
            "exit_code": (phases.get("quality_gate") or {}).get("exit_code"),
            "regression": (phases.get("quality_gate") or {}).get("regression"),
            "pytest": (phases.get("quality_gate") or {}).get("pytest"),
            "inputs_match": (phases.get("quality_gate") or {}).get("inputs_match"),
        },
    }
    if gate.returncode != 0:
        return _finish(report, EXIT_MISMATCH, "隔离副本里原样总门禁没有通过")

    after = {relative: _digest(PROJECT_ROOT / relative) for relative in FROZEN_REL}
    report["real_workspace_unchanged"] = before == after
    if before != after:
        return _finish(report, EXIT_MISMATCH, "真实工作目录的冻结文件发生了变化")
    return _finish(report, EXIT_OK, None)


def _finish(report: dict, exit_code: int, failure: str | None, report_dir: Path | None = None) -> int:
    report["expected"] = exit_code == EXIT_OK
    report["failure"] = failure
    directory = report_dir or (PROJECT_ROOT / STAGE_REL)
    try:
        write_report(report, directory / "isolated_report.json")
        print(f"报告：{directory.relative_to(PROJECT_ROOT).as_posix()}/isolated_report.json")
    except OSError:
        print("报告写入失败")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
