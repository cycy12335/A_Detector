"""CW-012 总门禁编排的区分性测试：阶段顺序、稳定退出码、失败关闭与报告安全。

这些用例刻意**不**真的跑全量 pytest（那由验收时的真实总门禁负责），而是用注入的
阶段替身把编排语义钉死：

- 静态失败时**绝不**运行 CW-009 阶段（失败确实阻止后续阶段）；
- CW-009 退出 0 但报告缺失/未标记通过时**仍然失败**（不因为"退出码好看"就变绿）；
- 超时、无法启动、工具失败各有稳定退出码；
- 报告只含安全字段：没有绝对路径、没有测试节点 ID、没有 stdout/stderr。

真实端到端由 `tests/integration/test_cw012_ci_inputs.py` 与验收时的
`python scripts/run_project_gate.py` 负责。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ai_detector.config import PROJECT_ROOT
from project_gate import gate
from project_gate.static_checks import (
    STATUS_PASSED,
    STATUS_TIMEOUT,
    STATUS_TOOL_ERROR,
    STATUS_VIOLATIONS,
    StaticOutcome,
)


def _static(status: str = STATUS_PASSED, *, violations: int = 0, by_code=None) -> StaticOutcome:
    return StaticOutcome(
        status=status,
        exit_code=0 if status == STATUS_PASSED else 1,
        duration_s=0.5,
        violations=violations,
        by_code=dict(by_code or {}),
        rules=("E9",),
        paths=("tests",),
        detail="python -m ruff check --isolated",
    )


def _summary(*, status: str | None = "passed", present: bool = True) -> dict:
    return {
        "report_present": present,
        "cw009_status": status,
        "cw009_exit_code": 0,
        "regression": {"cases_passed": 12, "cases_total": 12,
                       "max_score_absolute_deviation": 0.0, "deterministic": True},
        "pytest": {"status": "passed", "exit_code": 0, "counts": {"passed": 1756, "skipped": 3, "failed": 0}},
    }


# --- 稳定退出码 ----------------------------------------------------------


def test_exit_codes_are_distinct_and_documented() -> None:
    codes = [
        gate.EXIT_OK,
        gate.EXIT_STATIC_VIOLATIONS,
        gate.EXIT_STATIC_TOOL,
        gate.EXIT_STATIC_TIMEOUT,
        gate.EXIT_QUALITY_GATE,
        gate.EXIT_REPORT,
    ]
    assert len(set(codes)) == len(codes), "各阶段必须有互不相同的稳定退出码"
    assert gate.EXIT_OK == 0


# --- 阶段顺序与失败关闭 --------------------------------------------------


def test_static_violations_block_the_quality_gate_stage() -> None:
    """静态阶段失败时，CW-009 阶段**一次都不运行**。"""
    called: list[list[str]] = []

    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(STATUS_VIOLATIONS, violations=7),
        quality_gate_runner=lambda command, *, cwd, timeout_s: (called.append(command), (0, 1.0))[1],
        summary_loader=lambda path: _summary(),
    )
    assert called == [], "静态失败后不得继续运行质量门禁"
    assert outcome.status == gate.STATUS_FAILED
    assert outcome.exit_code == gate.EXIT_STATIC_VIOLATIONS
    assert outcome.error_code == "static_violations"


@pytest.mark.parametrize(
    "status,expected_exit",
    [(STATUS_TOOL_ERROR, gate.EXIT_STATIC_TOOL), (STATUS_TIMEOUT, gate.EXIT_STATIC_TIMEOUT)],
)
def test_static_tool_failures_have_their_own_exit_codes(status: str, expected_exit: int) -> None:
    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(status),
        quality_gate_runner=lambda command, *, cwd, timeout_s: pytest.fail("不应运行"),
        summary_loader=lambda path: _summary(),
    )
    assert outcome.exit_code == expected_exit
    assert outcome.error_code == f"static_{status}"


def test_passing_static_runs_the_quality_gate_stage_exactly_once() -> None:
    calls: list[tuple[list[str], Path, float]] = []

    def runner(command, *, cwd, timeout_s):
        calls.append((command, cwd, timeout_s))
        return 0, 12.5

    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(),
        quality_gate_runner=runner,
        summary_loader=lambda path: _summary(),
    )
    assert len(calls) == 1, "全量 pytest 只能跑一次"
    command, cwd, timeout_s = calls[0]
    assert command[1:] == [
        "scripts/run_quality_gate.py", "--config", "configs/quality_gate.yaml"
    ]
    assert cwd == PROJECT_ROOT, "必须在项目根运行"
    assert timeout_s > 1800.0, "外层超时必须宽于 CW-009 自己的 pytest 超时"
    assert outcome.status == gate.STATUS_PASSED and outcome.exit_code == 0


def test_quality_gate_failure_has_a_stable_exit_code() -> None:
    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(),
        quality_gate_runner=lambda command, *, cwd, timeout_s: (4, 30.0),
        summary_loader=lambda path: _summary(status="failed"),
    )
    assert outcome.exit_code == gate.EXIT_QUALITY_GATE
    assert outcome.error_code == "quality_gate_failed"
    assert outcome.quality_gate["exit_code"] == 4


def test_quality_gate_timeout_is_a_stable_failure() -> None:
    def runner(command, *, cwd, timeout_s):
        raise subprocess.TimeoutExpired(command, timeout_s)

    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(),
        quality_gate_runner=runner,
        summary_loader=lambda path: _summary(present=False, status=None),
    )
    assert outcome.exit_code == gate.EXIT_QUALITY_GATE
    assert outcome.error_code == "quality_gate_timeout"
    assert outcome.quality_gate["status"] == "timeout"


def test_quality_gate_that_cannot_start_is_a_stable_failure() -> None:
    def runner(command, *, cwd, timeout_s):
        raise OSError("cannot spawn")

    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(),
        quality_gate_runner=runner,
        summary_loader=lambda path: _summary(present=False, status=None),
    )
    assert outcome.exit_code == gate.EXIT_QUALITY_GATE
    assert outcome.error_code == "quality_gate_not_started"


# --- 退出码好看但报告不对：仍然失败 --------------------------------------


def test_exit_zero_without_a_report_still_fails() -> None:
    """CW-009 退出 0 但报告缺失：不得据此判通过（报告是唯一事实来源）。"""
    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(),
        quality_gate_runner=lambda command, *, cwd, timeout_s: (0, 5.0),
        summary_loader=lambda path: _summary(present=False, status=None),
    )
    assert outcome.status == gate.STATUS_FAILED
    assert outcome.error_code == "quality_gate_report_missing"


def test_exit_zero_with_a_non_passing_report_still_fails() -> None:
    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(),
        quality_gate_runner=lambda command, *, cwd, timeout_s: (0, 5.0),
        summary_loader=lambda path: _summary(status="failed"),
    )
    assert outcome.status == gate.STATUS_FAILED
    assert outcome.error_code == "quality_gate_report_mismatch"


def test_real_cw009_summary_reader_fails_closed_when_the_report_is_absent(
    tmp_path: Path,
) -> None:
    summary = gate._load_quality_gate_summary(tmp_path / "absent.json")
    assert summary["report_present"] is False
    assert summary["cw009_status"] is None


def test_real_cw009_summary_reader_survives_a_corrupt_report(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not json", encoding="utf-8")
    summary = gate._load_quality_gate_summary(path)
    assert summary["report_present"] is False


# --- 报告安全与原子写入 --------------------------------------------------


def test_report_carries_the_required_fields_without_leaking() -> None:
    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(STATUS_PASSED, violations=0),
        quality_gate_runner=lambda command, *, cwd, timeout_s: (0, 9.0),
        summary_loader=lambda path: _summary(),
    )
    report = gate.build_report(outcome)
    assert report["work_package"] == "CW-012"
    assert report["status"] == "passed"
    assert report["command_shape"] == "python scripts/run_project_gate.py"
    assert report["phases"][gate.PHASE_STATIC]["violations"] == 0
    quality = report["phases"][gate.PHASE_QUALITY_GATE]
    assert quality["regression"]["cases_total"] == 12
    assert quality["pytest"]["counts"]["passed"] == 1756
    assert quality["command_shape"] == (
        "python scripts/run_quality_gate.py --config configs/quality_gate.yaml"
    )
    assert set(report["environment"]) >= {"python", "ruff", "pytest"}

    rendered = json.dumps(report, ensure_ascii=False)
    assert str(PROJECT_ROOT) not in rendered, "报告不得含绝对路径"
    assert ".py::" not in rendered, "报告不得含 pytest 节点 ID"
    assert "stdout" not in rendered and "stderr" not in rendered
    assert "Traceback" not in rendered


def test_static_violation_counts_reach_the_report() -> None:
    outcome = gate.run_project_gate(
        root=PROJECT_ROOT,
        static_runner=lambda **kwargs: _static(
            STATUS_VIOLATIONS, violations=3, by_code={"F401": 2, "F821": 1}
        ),
        quality_gate_runner=lambda command, *, cwd, timeout_s: pytest.fail("不应运行"),
        summary_loader=lambda path: _summary(),
    )
    report = gate.build_report(outcome)
    static = report["phases"][gate.PHASE_STATIC]
    assert static["violations"] == 3
    assert static["by_code"] == {"F401": 2, "F821": 1}
    assert report["phases"][gate.PHASE_QUALITY_GATE] is None


def test_report_write_is_atomic_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"
    gate.write_report({"status": "passed"}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "passed"}
    assert [p.name for p in target.parent.iterdir()] == ["report.json"]


def test_report_write_failure_returns_the_report_exit_code(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "run_project_gate", lambda: gate.GateOutcome("passed", 0, None))
    monkeypatch.setattr(gate, "print_summary", lambda outcome: None)

    def boom(report, path):
        raise OSError("read-only")

    monkeypatch.setattr(gate, "write_report", boom)
    assert gate.main(["--report", str(tmp_path / "r.json")]) == gate.EXIT_REPORT


def test_main_returns_the_outcome_exit_code(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        gate, "run_project_gate", lambda: gate.GateOutcome("failed", gate.EXIT_STATIC_VIOLATIONS, "static_violations", _static(STATUS_VIOLATIONS, violations=1))
    )
    monkeypatch.setattr(gate, "print_summary", lambda outcome: None)
    assert gate.main(["--report", str(tmp_path / "r.json")]) == gate.EXIT_STATIC_VIOLATIONS
    assert (tmp_path / "r.json").is_file(), "失败时也要尽力留下报告"
