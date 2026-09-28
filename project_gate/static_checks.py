"""CW-012 静态检查阶段：用**冻结规则集**调用 ruff，并折叠成稳定的分类结果。

## 规则集为什么是"正确性 + 导入卫生"这六族

本包只选**能指向真实缺陷**的规则，不选风格规则：

===========  ==========================================================
规则族        抓什么
===========  ==========================================================
``E9``        语法错误——文件根本不能编译
``F63``      无效转义序列、对元组做断言、可疑比较
``F7``       语句位置错误（循环外 ``break``/``continue`` 等）
``F82``      未定义名称（含 ``__all__`` 与局部变量先引用后赋值）
``F811``     未使用名称被重复定义（遮蔽与死代码）
``F401``     未使用的导入（重构残留）
===========  ==========================================================

刻意**不**包含风格/现代化规则（``I``/``UP``/``SIM``/``B``/``RUF``）：实测它们在本仓库
会产生大量与语言（中文注释触发 ``RUF001`` 系列的歧义 Unicode 告警）或既有写法有关的
噪声，把门禁淹没在无关差异里。这里没有 ``ignore``、没有 per-file ignores、没有 ``noqa``
兜底——规则集本身就是**完整**选择集。

## 提交给 ruff 的路径是项目相对的

因此诊断里的文件位置也是相对的，不会把项目绝对路径带进报告或控制台。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

#: 冻结的规则选择集。**权威来源**：命令行上显式传入，不依赖任何配置文件，
#: 因此本仓库或用户的 ruff 配置改动都不会放宽它。与 ``pyproject.toml`` 的
#: ``[tool.ruff.lint] select`` 由测试断言保持一致。
FROZEN_RULES: tuple[str, ...] = ("E9", "F63", "F7", "F82", "F811", "F401")

#: 被检查的范围：验收要求的四个目录，加上本包新增的门禁包与仓库根 ``conftest.py``。
STATIC_PATHS: tuple[str, ...] = (
    "src/ai_detector",
    "scripts",
    "performance",
    "project_gate",
    "tests",
    "conftest.py",
)

#: 静态阶段的有限超时（秒）。ruff 在本仓库上不到 1 秒，这个上限只用于兜底挂死。
STATIC_TIMEOUT_S = 180.0

#: 结果分类（稳定字符串，进报告与控制台）。
STATUS_PASSED = "passed"
STATUS_VIOLATIONS = "violations"
STATUS_TOOL_ERROR = "tool_error"
STATUS_TIMEOUT = "timeout"


@dataclass(frozen=True)
class StaticOutcome:
    """一次静态检查的**安全**结果：只有计数与分类，没有文件内容或原始输出。"""

    status: str
    exit_code: int | None
    duration_s: float
    violations: int
    by_code: dict[str, int]
    rules: tuple[str, ...]
    paths: tuple[str, ...]
    detail: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASSED

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_s": self.duration_s,
            "violations": self.violations,
            "by_code": dict(sorted(self.by_code.items())),
            "rules": list(self.rules),
            "paths": list(self.paths),
            "detail": self.detail,
        }


def build_static_command(
    *, executable: str, paths: Sequence[str] = STATIC_PATHS, rules: Sequence[str] = FROZEN_RULES
) -> list[str]:
    """构造 ruff 命令：不经 shell，规则显式传入。

    ``--no-cache`` 保证结果不受缓存影响（``.ruff_cache`` 本身也被 Git 忽略）；
    ``--isolated`` 让 ruff 忽略任何 pyproject/用户级配置，只按这里的 ``--select`` 判定，
    因此"规则冻结"不依赖工作区状态。
    """
    return [
        executable, "-m", "ruff", "check",
        "--no-cache", "--isolated",
        "--select", ",".join(rules),
        "--output-format", "json",
        *paths,
    ]


def _safe_shape(command: Sequence[str]) -> str:
    """命令的**安全形状**：丢掉解释器绝对路径，只留模块与参数。"""
    parts = [part for part in command]
    if parts and (os.path.isabs(parts[0]) or Path(parts[0]).is_absolute()):
        parts = ["python", *parts[1:]]
    return " ".join(parts)


def _execute(command: list[str], *, cwd: Path, timeout_s: float) -> tuple[int, str]:
    """运行子进程并回收；返回 ``(退出码, stdout)``。

    Raises:
        subprocess.TimeoutExpired: 超时（调用方折叠为 ``timeout``）。
        OSError: 解释器或工具无法启动。
    """
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=flags,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _reap(process)
        raise
    return process.returncode, stdout


def _reap(process: subprocess.Popen) -> None:
    """超时后统一回收子进程；尽力而为，不掩盖原始超时。"""
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


def count_violations(stdout: str) -> tuple[int, dict[str, int]]:
    """从 ruff 的 JSON 输出里数出违规总数与按规则码的分布。

    只取 ``code`` 计数：**不**保留文件名、行号、消息或代码片段，因此报告里不会出现
    绝对路径、源码内容或任何载荷。输出不是合法 JSON（例如工具异常）时按 0 计，
    由退出码单独决定分类——计数缺失不会被当成"通过"。
    """
    try:
        payload = json.loads(stdout or "[]")
    except (json.JSONDecodeError, TypeError):
        return 0, {}
    if not isinstance(payload, list):
        return 0, {}
    by_code: dict[str, int] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        code = item.get("code")
        if isinstance(code, str) and code:
            by_code[code] = by_code.get(code, 0) + 1
    return sum(by_code.values()), by_code


def run_static_checks(
    *,
    root: Path,
    paths: Sequence[str] = STATIC_PATHS,
    rules: Sequence[str] = FROZEN_RULES,
    timeout_s: float = STATIC_TIMEOUT_S,
    executable: str | None = None,
    execute: Callable[..., tuple[int, str]] = _execute,
) -> StaticOutcome:
    """运行静态检查并折叠成稳定分类。

    Raises 一律不外传：启动失败、超时与工具异常都变成**失败**分类（``tool_error`` /
    ``timeout``），绝不静默变绿。

    Returns:
        :class:`StaticOutcome`；``status`` 为 ``passed`` 当且仅当 ruff 退出 0
        **且**没有违规计数。
    """
    command = build_static_command(
        executable=executable or sys.executable, paths=paths, rules=rules
    )
    shape = _safe_shape(command)
    started = time.monotonic()
    try:
        exit_code, stdout = execute(command, cwd=root, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return StaticOutcome(
            status=STATUS_TIMEOUT,
            exit_code=None,
            duration_s=round(time.monotonic() - started, 3),
            violations=0,
            by_code={},
            rules=tuple(rules),
            paths=tuple(paths),
            detail=shape,
        )
    except OSError:
        return StaticOutcome(
            status=STATUS_TOOL_ERROR,
            exit_code=None,
            duration_s=round(time.monotonic() - started, 3),
            violations=0,
            by_code={},
            rules=tuple(rules),
            paths=tuple(paths),
            detail=shape,
        )
    violations, by_code = count_violations(stdout)
    if exit_code == 0 and violations == 0:
        status = STATUS_PASSED
    elif exit_code == 1:
        # ruff 用 1 表示"发现了违规"。
        status = STATUS_VIOLATIONS
    else:
        # 2 表示 ruff 自身报错（参数/内部错误），其它非零同样按工具失败处理。
        status = STATUS_TOOL_ERROR
    return StaticOutcome(
        status=status,
        exit_code=exit_code,
        duration_s=round(time.monotonic() - started, 3),
        violations=violations,
        by_code=by_code,
        rules=tuple(rules),
        paths=tuple(paths),
        detail=shape,
    )
