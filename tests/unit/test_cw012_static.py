"""CW-012 静态检查阶段的区分性测试。

覆盖验收要求的三类区分性证据，且都用**真实 ruff**跑一遍，而不是只对着替身断言：

- 合法代码**通过**；
- 未定义名称（``F821``）与语法错误**失败**；
- 工具非零退出、无法启动、超时**都失败**（不会因为吞掉异常而变绿）。

另外守住两条"冻结"性质：命令里没有 shell、规则显式传入且 ``--isolated``；
``pyproject.toml`` 的 ``[tool.ruff.lint] select`` 与代码里的 :data:`FROZEN_RULES`
逐项相等，避免两处漂移。

ruff 是本包的可选依赖；未安装时本模块整体跳过（与既有 ``pytest.importorskip`` 用法一致），
静态阶段本身仍会在总门禁里**失败关闭**，不会被误当成通过。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_detector.config import PROJECT_ROOT
from project_gate.static_checks import (
    FROZEN_RULES,
    STATIC_PATHS,
    STATUS_PASSED,
    STATUS_TIMEOUT,
    STATUS_TOOL_ERROR,
    STATUS_VIOLATIONS,
    build_static_command,
    count_violations,
    run_static_checks,
)


def _ruff_available() -> bool:
    """ruff 以二进制 wheel 分发，没有可导入的模块，因此用 ``-m ruff --version`` 探测。"""
    try:
        return subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 环境级兜底
        return False


pytestmark = pytest.mark.skipif(
    not _ruff_available(), reason="需要 static extra：pip install -e \".[static]\""
)


def _write(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


# --- 规则集与命令形状 ----------------------------------------------------


def test_frozen_rules_are_exactly_what_pyproject_declares() -> None:
    """两处规则集必须逐项相等：代码里的常量是权威来源，pyproject 供手工运行。"""
    import tomllib

    raw = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = raw["tool"]["ruff"]["lint"]["select"]
    assert tuple(declared) == FROZEN_RULES


def test_static_scope_covers_the_required_directories() -> None:
    """验收要求的四个目录必须都在检查范围内。"""
    for required in ("src/ai_detector", "scripts", "performance", "tests"):
        assert required in STATIC_PATHS


def test_command_is_frozen_isolated_and_shell_free() -> None:
    command = build_static_command(executable="python")
    assert command[:3] == ["python", "-m", "ruff"]
    assert "--isolated" in command, "必须忽略工作区与用户级配置，规则只由命令行决定"
    assert "--no-cache" in command
    assert command[command.index("--select") + 1] == ",".join(FROZEN_RULES)
    assert "shell" not in " ".join(command)


def test_reported_command_shape_hides_the_interpreter_path() -> None:
    """报告里的命令形状不得带解释器绝对路径。"""
    from project_gate.static_checks import _safe_shape

    shape = _safe_shape(build_static_command(executable=str(PROJECT_ROOT / ".venv/python.exe")))
    assert shape.startswith("python -m ruff")
    assert str(PROJECT_ROOT) not in shape


# --- 真实 ruff：合法代码通过 --------------------------------------------


def test_legal_code_passes_under_the_frozen_rules(tmp_path: Path) -> None:
    _write(tmp_path, "legal.py", "def add(left: int, right: int) -> int:\n    return left + right\n")
    outcome = run_static_checks(root=tmp_path, paths=("legal.py",))
    assert outcome.status == STATUS_PASSED
    assert outcome.violations == 0 and outcome.exit_code == 0


def test_the_repository_itself_is_clean_under_the_frozen_rules() -> None:
    """本仓库在冻结规则下必须零违规——这是静态阶段真正承重的证据。"""
    outcome = run_static_checks(root=PROJECT_ROOT)
    assert outcome.status == STATUS_PASSED, outcome.as_dict()
    assert outcome.violations == 0
    assert outcome.by_code == {}


# --- 真实 ruff：未定义名称与语法错误失败 --------------------------------


def test_undefined_name_is_reported_as_f821(tmp_path: Path) -> None:
    _write(tmp_path, "undef.py", "def render():\n    return missing_helper()\n")
    outcome = run_static_checks(root=tmp_path, paths=("undef.py",))
    assert outcome.status == STATUS_VIOLATIONS
    assert outcome.exit_code == 1
    assert outcome.by_code.get("F821") == 1


def test_syntax_error_fails_even_though_the_file_cannot_be_parsed(tmp_path: Path) -> None:
    """语法错误也必须被抓住（ruff 0.14 把它的诊断码报成 ``invalid-syntax``）。"""
    _write(tmp_path, "broken.py", "def render(:\n    pass\n")
    outcome = run_static_checks(root=tmp_path, paths=("broken.py",))
    assert outcome.status == STATUS_VIOLATIONS
    assert outcome.violations >= 1
    assert "invalid-syntax" in outcome.by_code


def test_unused_import_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "unused.py", "import json\n")
    outcome = run_static_checks(root=tmp_path, paths=("unused.py",))
    assert outcome.status == STATUS_VIOLATIONS
    assert outcome.by_code.get("F401") == 1


# --- 工具失败、超时、无法启动 -------------------------------------------


def test_tool_nonzero_exit_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    """ruff 自身报错（退出码 2）必须折叠为 ``tool_error``，不得当成通过。"""
    def fake_execute(command, *, cwd, timeout_s):
        return 2, ""

    outcome = run_static_checks(root=tmp_path, paths=("x.py",), execute=fake_execute)
    assert outcome.status == STATUS_TOOL_ERROR
    assert outcome.exit_code == 2
    assert not outcome.passed


def test_timeout_is_a_failure(tmp_path: Path) -> None:
    def fake_execute(command, *, cwd, timeout_s):
        raise subprocess.TimeoutExpired(command, timeout_s)

    outcome = run_static_checks(root=tmp_path, paths=("x.py",), execute=fake_execute)
    assert outcome.status == STATUS_TIMEOUT
    assert not outcome.passed


def test_unstartable_tool_is_a_failure(tmp_path: Path) -> None:
    def fake_execute(command, *, cwd, timeout_s):
        raise FileNotFoundError("no interpreter")

    outcome = run_static_checks(root=tmp_path, paths=("x.py",), execute=fake_execute)
    assert outcome.status == STATUS_TOOL_ERROR
    assert not outcome.passed


def test_a_missing_ruff_installation_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    """真实场景：static extra 未安装。总门禁必须失败关闭而不是跳过。"""
    outcome = run_static_checks(
        root=tmp_path, paths=("x.py",), executable=str(tmp_path / "definitely-not-python")
    )
    assert outcome.status == STATUS_TOOL_ERROR
    assert not outcome.passed


def test_violation_with_unparseable_output_still_fails(tmp_path: Path) -> None:
    """退出 1 但输出不是 JSON：计数缺失，但分类仍必须是违规（不能变绿）。"""
    def fake_execute(command, *, cwd, timeout_s):
        return 1, "not json at all"

    outcome = run_static_checks(root=tmp_path, paths=("x.py",), execute=fake_execute)
    assert outcome.status == STATUS_VIOLATIONS
    assert outcome.violations == 0, "解析不到计数时按 0 记，但分类由退出码决定"
    assert not outcome.passed


# --- 计数只保留规则码 ----------------------------------------------------


def test_counts_are_grouped_by_rule_code() -> None:
    payload = json.dumps([
        {"code": "F401", "filename": "a.py"},
        {"code": "F401", "filename": "b.py"},
        {"code": "F821", "filename": "c.py"},
    ])
    total, by_code = count_violations(payload)
    assert total == 3
    assert by_code == {"F401": 2, "F821": 1}


def test_counts_discard_filenames_and_messages() -> None:
    """诊断里的绝对路径与消息**不得**进入结果；只留规则码计数。"""
    absolute = str(PROJECT_ROOT / "src" / "ai_detector" / "service" / "app.py")
    payload = json.dumps([
        {"code": "F401", "filename": absolute, "message": "`os` imported but unused",
         "location": {"row": 12, "column": 1}},
    ])
    outcome = run_static_checks(
        root=PROJECT_ROOT,
        paths=("src/ai_detector",),
        execute=lambda command, *, cwd, timeout_s: (1, payload),
    )
    rendered = json.dumps(outcome.as_dict(), ensure_ascii=False)
    assert str(PROJECT_ROOT) not in rendered
    assert "app.py" not in rendered
    assert "imported but unused" not in rendered
    assert outcome.by_code == {"F401": 1}


@pytest.mark.parametrize("payload", ["", "[]", "{}", "null", "[1, 2]", "not json"])
def test_degenerate_output_is_treated_as_no_counts(payload: str) -> None:
    assert count_violations(payload) == (0, {})
