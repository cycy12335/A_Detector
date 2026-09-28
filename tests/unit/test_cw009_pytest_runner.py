"""CW-009 pytest 阶段：子进程形状、计数解析、失败/超时/无法启动与回收。

门禁的最后一环必须"跑得动、停得下、收得回"。这里用**真实子进程**覆盖超时与无法启动
两条路径（fake 只能证明分支被走到，证明不了 kill 之后确实收尸了）。
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

from ai_detector.config import PROJECT_ROOT
from ai_detector.quality.config import FROZEN_PYTEST_ARGUMENTS, QualityGateConfig
from ai_detector.quality.gate import (
    PYTEST_CWD_PLACEHOLDER,
    PYTEST_PROGRAM_PLACEHOLDER,
    PytestOutcome,
    build_pytest_command,
    run_pytest,
)

#: 子进程睡眠时长与门禁超时：超时必须远小于睡眠，才能证明"是 kill 生效而不是等完了"。
SLEEP_SECONDS = 45
TIMEOUT_SECONDS = 1.0


def _sleeping_command() -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({SLEEP_SECONDS})"]


def test_default_command_is_current_interpreter_module_call() -> None:
    config = QualityGateConfig.from_yaml("configs/quality_gate.yaml")
    command = build_pytest_command(config)

    assert command == [sys.executable, "-m", "pytest"]
    assert command[1:] == list(FROZEN_PYTEST_ARGUMENTS)
    assert sys.executable in command


def test_success_parses_counts_without_keeping_output() -> None:
    script = "print('1 failed, 7 passed, 2 skipped, 3 warnings in 0.42s')"
    outcome = run_pytest(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, timeout_seconds=60.0
    )

    assert outcome.status == "passed"
    assert outcome.exit_code == 0
    assert (outcome.passed, outcome.skipped, outcome.failed) == (7, 2, 1)
    assert outcome.child_reaped is True
    # 结果对象里没有任何"输出"字段：pytest 的 stdout/stderr 绝不外传。
    assert {field.name for field in dataclasses.fields(PytestOutcome)} == {
        "status",
        "exit_code",
        "duration_seconds",
        "passed",
        "skipped",
        "failed",
        "errors",
        "child_reaped",
    }


def test_error_count_is_parsed() -> None:
    script = "print('2 errors in 0.10s')"
    outcome = run_pytest(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, timeout_seconds=60.0
    )
    assert outcome.errors == 2


def test_failure_exit_code_is_preserved() -> None:
    script = "import sys; print('3 failed, 4 passed in 0.20s'); sys.exit(1)"
    outcome = run_pytest(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, timeout_seconds=60.0
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 1
    assert (outcome.passed, outcome.failed) == (4, 3)


def test_timeout_kills_and_reaps_the_child() -> None:
    started = time.perf_counter()
    outcome = run_pytest(
        _sleeping_command(), cwd=PROJECT_ROOT, timeout_seconds=TIMEOUT_SECONDS
    )
    elapsed = time.perf_counter() - started

    assert outcome.status == "timeout"
    assert outcome.exit_code is None
    assert outcome.child_reaped is True, "超时后子进程必须被可靠回收"
    assert elapsed < SLEEP_SECONDS / 2, "超时必须真的 kill，而不是等子进程自己结束"
    assert (outcome.passed, outcome.skipped, outcome.failed, outcome.errors) == (0, 0, 0, 0)


def test_spawn_failure_is_reported_instead_of_raised() -> None:
    outcome = run_pytest(
        ["cw009_definitely_not_an_executable"], cwd=PROJECT_ROOT, timeout_seconds=10.0
    )

    assert outcome.status == "spawn_failed"
    assert outcome.exit_code is None
    assert outcome.child_reaped is True


def test_counts_ignore_unrelated_numbers_in_the_tail() -> None:
    script = "print('collected 1346 items'); print('1346 passed, 3 skipped in 45.6s')"
    outcome = run_pytest(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, timeout_seconds=60.0
    )
    assert (outcome.passed, outcome.skipped) == (1346, 3)


def test_report_command_shape_never_leaks_the_interpreter_path() -> None:
    assert not Path(PYTEST_PROGRAM_PLACEHOLDER).is_absolute()
    assert not Path(PYTEST_CWD_PLACEHOLDER).is_absolute()
