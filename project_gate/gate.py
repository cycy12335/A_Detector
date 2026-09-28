"""CW-012 本地总质量入口的编排与安全报告。

``python scripts/run_project_gate.py`` 按固定顺序运行两个阶段：

1. **static**（:mod:`project_gate.static_checks`）：用冻结规则集跑 ruff；
2. **quality_gate**：原样执行 CW-009 的默认命令
   ``python scripts/run_quality_gate.py --config configs/quality_gate.yaml``，
   即完整的 preflight → regression → pytest 链路。

## 为什么第二阶段是"真跑命令"而不是导入函数

复用当前的 ``sys.executable`` 以子进程方式执行**原样命令**，保证链路与人工执行完全一致、
只跑一次全量 pytest、且阶段之间互不污染（CW-009 的退出码语义保持冻结）。
报告里的 CW-009 计数从它自己写的报告文件里读，不重复计算、不另跑一遍。

## 退出码（稳定）

===  ==================================================
码    含义
===  ==================================================
0    两个阶段都通过
2    静态阶段发现违规
3    静态工具失败（无法启动 / 自身报错）
4    静态阶段超时
5    CW-009 质量门禁失败（其自身退出码记在报告里）
6    报告写入失败
===  ==================================================

## 报告只写安全信息

阶段、命令**形状**（丢解释器绝对路径）、退出码、耗时、违规**计数**（不写文件名与行号）、
CW-009 报告里的状态与计数、以及工具版本。不写完整 stdout/stderr、测试节点 ID、
请求/特征值、密钥、绝对路径、bundle 内容或数据库行。写入是原子的，失败时尽力留下报告。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from project_gate.static_checks import (
    STATIC_PATHS,
    STATIC_TIMEOUT_S,
    StaticOutcome,
    run_static_checks,
)

#: 项目根：本文件位于 ``<root>/project_gate/gate.py``。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: CW-009 的默认命令（相对项目根，逐字对应 README 与 CW-009 验收记录）。
QUALITY_GATE_SCRIPT_REL = "scripts/run_quality_gate.py"
QUALITY_GATE_CONFIG_REL = "configs/quality_gate.yaml"

#: CW-009 报告路径的兜底值；正常情况从门禁配置的 ``report.path`` 读取。
QUALITY_GATE_REPORT_REL = "artifacts/reports/cw009_quality_gate/report.json"

#: 本包报告（继续被 ``.gitignore`` 的 ``artifacts/**`` 规则排除）。
REPORT_REL = "artifacts/reports/cw012_project_gate/report.json"

#: 质量门禁阶段的有限超时（秒）。CW-009 自己的 pytest 超时是 1800 s，这里必须更宽，
#: 否则外层会先杀掉它、把一次"pytest 超时"（CW-009 有专属退出码 5）误报成总门禁超时。
QUALITY_GATE_TIMEOUT_S = 2700.0

#: 稳定退出码。
EXIT_OK = 0
EXIT_STATIC_VIOLATIONS = 2
EXIT_STATIC_TOOL = 3
EXIT_STATIC_TIMEOUT = 4
EXIT_QUALITY_GATE = 5
EXIT_REPORT = 6

#: 命令形状（不含解释器绝对路径）。
COMMAND_SHAPE = "python scripts/run_project_gate.py"

#: 结果分类（稳定字符串）。
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"

PHASE_STATIC = "static"
PHASE_QUALITY_GATE = "quality_gate"

#: 静态阶段分类 → 总门禁退出码。
STATIC_EXIT_CODES = {
    "violations": EXIT_STATIC_VIOLATIONS,
    "tool_error": EXIT_STATIC_TOOL,
    "timeout": EXIT_STATIC_TIMEOUT,
}


@dataclass
class GateOutcome:
    """总门禁结果。"""

    status: str
    exit_code: int
    error_code: str | None
    static: StaticOutcome | None = None
    quality_gate: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASSED


def _environment() -> dict[str, str]:
    """工具版本；缺失记为 ``missing``（本机环境特征，不含路径）。"""
    result = {"python": sys.version.split()[0]}
    for name in ("ruff", "pytest"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "missing"
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reap(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:  # pragma: no cover - 平台级兜底
            pass


def build_quality_gate_command(*, executable: str, config_rel: str = QUALITY_GATE_CONFIG_REL) -> list[str]:
    """CW-009 阶段的命令：与人工执行的默认命令逐字一致。"""
    return [executable, QUALITY_GATE_SCRIPT_REL, "--config", config_rel]


def quality_gate_report_path(root: Path, config_rel: str = QUALITY_GATE_CONFIG_REL) -> Path:
    """从门禁配置的 ``report.path`` 解析 CW-009 报告位置（项目相对）。

    解析不了就回落到默认路径：报告缺失会让质量门禁阶段失败关闭，不会变绿。
    """
    try:
        raw = yaml.safe_load((root / config_rel).read_text(encoding="utf-8"))
        declared = raw["quality_gate"]["report"]["path"]
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError):
        return root / QUALITY_GATE_REPORT_REL
    if not isinstance(declared, str) or not declared:
        return root / QUALITY_GATE_REPORT_REL
    return root / declared


def _execute_quality_gate(
    command: list[str], *, cwd: Path, timeout_s: float
) -> tuple[int, float]:
    """运行 CW-009 命令并回收进程，返回 ``(退出码, 墙钟耗时)``。

    子进程输出直接丢弃：总门禁不需要它，也就不可能把它写进报告。
    """
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    try:
        exit_code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _reap(process)
        raise
    return exit_code, round(time.monotonic() - started, 3)


def _load_quality_gate_summary(report_path: Path) -> dict[str, Any]:
    """从 CW-009 自己的报告里取**安全**摘要；缺失或不可解析时为 ``None`` 字段。

    刻意不重新计算任何统计：报告是唯一事实来源，总门禁只做转述。
    字段名按 ``cw009_quality_gate_report_v1`` 的**实际**键名读取
    （``overall_status``、``max_score_absolute_deviation``），
    并要求四个冻结输入的 ``match`` 标志一并转述——它们是 CI 输入缺口的定位依据。
    """
    summary: dict[str, Any] = {
        "report_present": False,
        "cw009_status": None,
        "cw009_exit_code": None,
        "inputs_match": None,
        "regression": None,
        "pytest": None,
    }
    if not report_path.is_file():
        return summary
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return summary
    if not isinstance(payload, dict):
        return summary
    summary["report_present"] = True
    summary["cw009_status"] = payload.get("overall_status")
    summary["cw009_exit_code"] = payload.get("exit_code")
    inputs = payload.get("inputs")
    if isinstance(inputs, dict):
        summary["inputs_match"] = {
            str(label): bool(section.get("match"))
            for label, section in inputs.items()
            if isinstance(section, dict)
        }
    regression = payload.get("regression")
    if isinstance(regression, dict) and regression:
        summary["regression"] = {
            "cases_passed": regression.get("cases_passed"),
            "cases_total": regression.get("cases_total"),
            "cases_failed": regression.get("cases_failed"),
            "max_score_absolute_deviation": regression.get("max_score_absolute_deviation"),
            "deterministic": regression.get("deterministic"),
        }
    pytest_summary = payload.get("pytest")
    if isinstance(pytest_summary, dict) and pytest_summary:
        counts = pytest_summary.get("counts")
        summary["pytest"] = {
            "status": pytest_summary.get("status"),
            "exit_code": pytest_summary.get("exit_code"),
            "counts": dict(counts) if isinstance(counts, dict) else None,
        }
    return summary


def run_project_gate(
    *,
    root: Path = PROJECT_ROOT,
    config_rel: str = QUALITY_GATE_CONFIG_REL,
    timeout_s: float = QUALITY_GATE_TIMEOUT_S,
    static_timeout_s: float = STATIC_TIMEOUT_S,
    executable: str | None = None,
    static_runner: Callable[..., StaticOutcome] = run_static_checks,
    quality_gate_runner: Callable[..., tuple[int, float]] = _execute_quality_gate,
    summary_loader: Callable[[Path], dict[str, Any]] = _load_quality_gate_summary,
) -> GateOutcome:
    """按 static → quality_gate 顺序运行；任一阶段失败即失败关闭。

    Args:
        config_rel: CW-009 门禁配置的项目相对路径。默认就是验收要求的那一个；
            只有测试才会传别的值（用于在**不触碰真实产物**的前提下构造缺 bundle 的反例）。
    """
    python = executable or sys.executable
    report_path = quality_gate_report_path(root, config_rel)
    static = static_runner(
        root=root, paths=STATIC_PATHS, timeout_s=static_timeout_s, executable=python
    )
    if not static.passed:
        return GateOutcome(
            status=STATUS_FAILED,
            exit_code=STATIC_EXIT_CODES.get(static.status, EXIT_STATIC_TOOL),
            error_code=f"static_{static.status}",
            static=static,
        )

    command = build_quality_gate_command(executable=python, config_rel=config_rel)
    shape = f"python {QUALITY_GATE_SCRIPT_REL} --config {config_rel}"
    try:
        exit_code, duration = quality_gate_runner(command, cwd=root, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return GateOutcome(
            status=STATUS_FAILED,
            exit_code=EXIT_QUALITY_GATE,
            error_code="quality_gate_timeout",
            static=static,
            quality_gate={
                "status": "timeout",
                "exit_code": None,
                "duration_s": None,
                "command_shape": shape,
                **summary_loader(report_path),
            },
        )
    except OSError:
        return GateOutcome(
            status=STATUS_FAILED,
            exit_code=EXIT_QUALITY_GATE,
            error_code="quality_gate_not_started",
            static=static,
            quality_gate={
                "status": "tool_error",
                "exit_code": None,
                "duration_s": None,
                "command_shape": shape,
                **summary_loader(report_path),
            },
        )

    summary = summary_loader(report_path)
    passed = exit_code == 0 and summary.get("cw009_status") == "passed"
    quality_status = STATUS_PASSED if passed else STATUS_FAILED
    error_code = None
    if not passed:
        if exit_code != 0:
            error_code = "quality_gate_failed"
        else:
            # 退出 0 但报告缺失/未标记通过：报告与退出码不一致，按失败关闭。
            error_code = "quality_gate_report_missing" if not summary.get(
                "report_present"
            ) else "quality_gate_report_mismatch"
    return GateOutcome(
        status=STATUS_PASSED if passed else STATUS_FAILED,
        exit_code=EXIT_OK if passed else EXIT_QUALITY_GATE,
        error_code=error_code,
        static=static,
        quality_gate={
            "status": quality_status,
            "exit_code": exit_code,
            "duration_s": duration,
            "command_shape": shape,
            **summary,
        },
    )


def build_report(outcome: GateOutcome) -> dict[str, Any]:
    """构造机器可读报告（只含安全字段）。"""
    return {
        "work_package": "CW-012",
        "schema_version": "cw012_project_gate_report_v1",
        "created_at_utc": _utc_now(),
        "status": outcome.status,
        "exit_code": outcome.exit_code,
        "error_code": outcome.error_code,
        "command_shape": COMMAND_SHAPE,
        "environment": _environment(),
        "phases": {
            PHASE_STATIC: outcome.static.as_dict() if outcome.static else None,
            PHASE_QUALITY_GATE: outcome.quality_gate or None,
        },
        "limitations": (
            "本机单进程门禁。CI 输入审计见 docs/ci_readiness.md：当前无 Git 仓库与托管平台，"
            "已验收 bundle 被 .gitignore 排除且供给方式未确定，因此不得据此宣称 hosted CI 已运行。"
        ),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """原子写入报告：先写同目录临时文件再 ``os.replace``。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def print_summary(outcome: GateOutcome) -> None:
    """控制台摘要：只有分类、计数与阶段名，不含输出、路径或载荷。"""
    static = outcome.static
    if static is not None:
        print(f"static      {static.status}  violations={static.violations}")
        if static.by_code:
            for code, count in sorted(static.by_code.items()):
                print(f"            {code} x{count}")
    gate = outcome.quality_gate
    if gate:
        pytest_info = gate.get("pytest") or {}
        counts = pytest_info.get("counts") or {}
        print(
            f"quality_gate {gate.get('status')}  exit={gate.get('exit_code')}"
            f"  regression={gate.get('regression')}"
        )
        print(f"            pytest counts={counts}")
    print(f"status      {outcome.status}  exit={outcome.exit_code}  error={outcome.error_code}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_project_gate.py",
        description="CW-012 本地总门禁：静态检查 + CW-009 默认质量门禁。",
    )
    parser.add_argument(
        "--report",
        default=REPORT_REL,
        help=f"报告的项目相对路径（默认 {REPORT_REL}）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = Path(args.report)
    outcome = run_project_gate()
    print_summary(outcome)
    try:
        write_report(build_report(outcome), report_path)
    except OSError:
        print("报告写入失败（report_write_failed）")
        return EXIT_REPORT
    print(f"报告：{args.report}")
    return outcome.exit_code
