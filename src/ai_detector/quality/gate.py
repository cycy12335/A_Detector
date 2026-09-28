"""CW-009 一键质量门禁：预检 → 回归 → 完整 pytest → 安全报告。

## 顺序不可交换

1. **preflight**：先把四个输入的**二进制**实际 SHA-256 **全部收齐**并写进报告（存在性与
   摘要收集阶段**不解析任何内容**），再统一判定；只要有一个不符，就不 ``joblib.load``、
   不解析 YAML/JSON、不推理、不跑 pytest——直接以退出码 2 失败关闭。报告里的四个
   ``actual_sha256`` / ``match`` 因此是完整的，而失败原因也不会被"第一个不匹配"掩盖；
2. **regression**：用生产顺序求实际结果并与冻结期望比较（含双跑确定性）；
3. **pytest**：以 ``sys.executable -m pytest`` 子进程运行完整测试套件（固定 cwd 为项目根、
   有限超时、不使用 shell）。

## 退出码（冻结）

======== ==== ==================================================
退出码     含义
======== ==== ==================================================
0        全部通过
2        配置 / 预检 / 语料契约失败
3        回归或确定性失败
4        pytest 非零（含无法启动子进程）
5        pytest 超时
6        报告写入失败
======== ==== ==================================================

**未预期的内部错误也失败关闭**：它按发生阶段返回该阶段的退出码，报告里的 ``error_code``
为 ``internal_error``。报告异常只写异常**类别名**，绝不写 traceback 或消息正文（消息可能
带绝对路径）。

## 没有"放行开关"

命令行只有 ``--config`` 与 ``--regression-only``。没有跳过哈希、放宽容差、更新 oracle、
接受当前结果的参数——语料一旦偏离，唯一的出路是显式评审后重新生成语料并更新配置里的
冻结指纹，而不是在命令行上把它按下去。

同理，配置 YAML 与语料 JSON 在**任意层级**出现重复映射键都会被拒绝（相同值与冲突值
一视同仁）：解析器的"后者覆盖前者"会让评审看到的文本与实际采用的对象不一致，因此
门禁走的是自带重复键检测的严格加载器，而不是 ``yaml.safe_load`` / 普通 ``json.loads``。
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ai_detector import __version__
from ai_detector.config import PROJECT_ROOT
from ai_detector.quality.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORT_PATH,
    WORK_PACKAGE,
    QualityGateConfig,
    collect_frozen_digests,
    first_problem,
    resolve_within_project,
)
from ai_detector.quality.corpus import (
    RegressionCorpus,
    forbidden_report_tokens,
    load_corpus,
)
from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode
from ai_detector.quality.regression import (
    RegressionOutcome,
    aggregate_error_code,
    check_coverage,
    failed_fields_by_case,
    run_regression,
)
from ai_detector.quality.report import (
    REPORT_SCHEMA_VERSION,
    write_report,
)
from ai_detector.service.config import ServiceConfig, ServiceConfigError
from ai_detector.service.runtime import load_policy_engine, load_runtime

# --- 退出码（冻结） -----------------------------------------------------

EXIT_OK = 0
EXIT_PREFLIGHT = 2
EXIT_REGRESSION = 3
EXIT_PYTEST = 4
EXIT_PYTEST_TIMEOUT = 5
EXIT_REPORT = 6

#: 子进程被 kill 后等待回收的上限（秒）。
REAP_TIMEOUT_SECONDS = 30.0

#: pytest 计数的解析窗口：只看输出尾部，避免匹配到用例名里的数字。
_COUNT_WINDOW = 4000

_COUNT_PATTERNS = {
    "passed": re.compile(r"(\d+) passed"),
    "skipped": re.compile(r"(\d+) skipped"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) errors?"),
}

#: pytest 命令在报告里的安全形状：解释器只写成占位符，不写真实绝对路径。
PYTEST_PROGRAM_PLACEHOLDER = "<sys.executable>"
PYTEST_CWD_PLACEHOLDER = "<project_root>"

LIMITATIONS: tuple[str, ...] = (
    "固定回归集是 golden master：它只检测实现或产物是否偏离已验收基线，"
    "不证明模型正确、公平、稳定或适合真实风控，也不是在 test 集上重新评估效果。",
    "语料是纯合成输入，不含真实交易 ID、标签或原始数据行；期望值由当前已验收 "
    "bundle/policy 一次性只读求得，门禁没有任何 update/bless/record 模式。",
    "本包不包含 CI、容器、压力/并发测试、漂移监控与模型再训练。",
)


@dataclass(frozen=True)
class PytestOutcome:
    """一次 pytest 子进程调用的结果（不含任何被测输出）。"""

    status: str  # passed | failed | timeout | spawn_failed
    exit_code: int | None
    duration_seconds: float
    passed: int
    skipped: int
    failed: int
    errors: int
    #: 子进程是否已被回收（``poll()`` 有返回值）。超时分支上这是"kill 之后确实收尸了"的证据。
    child_reaped: bool = True


@dataclass(frozen=True)
class GateOutcome:
    """门禁的最终结果：退出码 + 报告路径 + 已写入的报告（写失败时为 ``None``）。"""

    exit_code: int
    report_path: str
    report: dict[str, Any] | None
    report_written: bool


# --- pytest 子进程 ------------------------------------------------------


def build_pytest_command(config: QualityGateConfig) -> list[str]:
    """默认 pytest 命令：**当前解释器** + 配置冻结的模块参数。"""
    return [sys.executable, *config.pytest_arguments]


def _parse_counts(output: str) -> dict[str, int]:
    """从 pytest 输出尾部解析通过/跳过/失败/错误计数。

    **只取尾部窗口并只保留计数**：完整 stdout 绝不进入报告（其中可能含用例名与路径）。
    """
    tail = output[-_COUNT_WINDOW:]
    counts: dict[str, int] = {}
    for name, pattern in _COUNT_PATTERNS.items():
        matches = pattern.findall(tail)
        counts[name] = int(matches[-1]) if matches else 0
    return counts


def _reap(process: "subprocess.Popen[str]") -> bool:
    """可靠回收子进程：先 kill，再 communicate 收尸。失败也绝不抛出。

    Returns:
        ``True`` 表示子进程确实已经被回收（``poll()`` 已能给出退出码）。
    """
    try:
        process.kill()
    except OSError:  # pragma: no cover - 进程已退出
        pass
    try:
        process.communicate(timeout=REAP_TIMEOUT_SECONDS)
    except (subprocess.TimeoutExpired, OSError, ValueError):  # pragma: no cover - 兜底
        pass
    return process.poll() is not None


def run_pytest(
    command: Sequence[str], *, cwd: Path, timeout_seconds: float
) -> PytestOutcome:
    """运行 pytest 子进程并折叠成一个 :class:`PytestOutcome`。

    固定 ``shell=False``、``stdin=DEVNULL``、合并 stderr 到 stdout（方便解析计数但不外传），
    超时后 kill 并回收。子进程无法启动时返回 ``spawn_failed``，不抛异常。
    """
    started = time.perf_counter()
    try:
        process = subprocess.Popen(  # noqa: S603 - 命令由配置与当前解释器构成，不经 shell
            list(command),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return _no_child_outcome("spawn_failed", started)

    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return _no_child_outcome("timeout", started, child_reaped=_reap(process))
    except (OSError, ValueError):  # pragma: no cover - 管道异常时同样失败关闭
        return _no_child_outcome("spawn_failed", started, child_reaped=_reap(process))

    counts = _parse_counts(output or "")
    return PytestOutcome(
        status="passed" if process.returncode == 0 else "failed",
        exit_code=process.returncode,
        duration_seconds=round(time.perf_counter() - started, 3),
        passed=counts["passed"],
        skipped=counts["skipped"],
        failed=counts["failed"],
        errors=counts["errors"],
        child_reaped=process.poll() is not None,
    )


def _no_child_outcome(
    status: str, started: float, *, child_reaped: bool = True
) -> PytestOutcome:
    """构造一个"没有产出计数"的结果（无法启动或超时）。"""
    return PytestOutcome(
        status=status,
        exit_code=None,
        duration_seconds=round(time.perf_counter() - started, 3),
        passed=0,
        skipped=0,
        failed=0,
        errors=0,
        child_reaped=child_reaped,
    )


# --- 预检 ---------------------------------------------------------------


def preflight(
    config: QualityGateConfig, report: dict[str, Any]
) -> tuple[Any, Any, RegressionCorpus]:
    """核对四个输入的**二进制**摘要，然后才做语义解析、反序列化与加载。

    顺序是硬边界，分两段：

    1. **先收齐**四个输入的存在性与实际 SHA-256，把每个实际存在的输入的 ``actual_sha256``
       与 ``match`` 都写进报告；任何一个未通过都只在这时**统一**失败关闭。期间不解析
       YAML/JSON、不 ``joblib.load``、不推理、不跑 pytest；
    2. 只有四个全部匹配，才开始语义解析、反序列化与加载。

    Returns:
        ``(runtime, policy_engine, corpus)``。

    Raises:
        QualityGateError: 任一输入缺失/类型不符/指纹不符、服务配置非法、bundle/政策加载
            失败，或配置与运行时/语料的契约不一致。
    """
    digests = collect_frozen_digests(
        (config.service_config, config.bundle, config.policy, config.corpus)
    )
    # 先登记到报告再判定：失败报告里四个存在输入的 actual_sha256 / match 必须是完整的。
    report["inputs"] = {digest.label: digest.as_report_entry() for digest in digests}

    problem = first_problem(digests)
    if problem is not None:
        raise QualityGateError(problem, "四个冻结输入的二进制预检未通过")

    paths = {
        digest.label: resolve_within_project(digest.relative_path, digest.label)
        for digest in digests
    }

    # --- 以下才允许做语义解析与反序列化 ---
    try:
        service_config = ServiceConfig.from_yaml(config.service_config.relative_path)
    except ServiceConfigError:
        raise QualityGateError(
            QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION, "服务配置解析失败"
        ) from None

    try:
        runtime = load_runtime(service_config)
    except Exception as error:  # noqa: BLE001 - 只记稳定分类码，不外泄路径与内容
        raise QualityGateError(
            QualityGateErrorCode.RUNTIME_LOAD_FAILED,
            f"bundle 加载失败：{_stable_code(error)}",
        ) from None

    try:
        engine = load_policy_engine(service_config)
    except Exception as error:  # noqa: BLE001 - 同上
        raise QualityGateError(
            QualityGateErrorCode.RUNTIME_LOAD_FAILED, f"政策加载失败：{_stable_code(error)}"
        ) from None

    corpus = load_corpus(
        paths["corpus"],
        source_features=runtime.source_features,
        categorical_features=runtime.categorical_features,
    )

    config.check_runtime_agreement(runtime, engine)
    config.check_corpus_agreement(corpus)

    report["versions"] = {
        "package_version": __version__,
        "model_version": str(runtime.model_version),
        "policy_version": str(engine.policy_version),
        "score_semantics": str(runtime.score_semantics),
        "threshold_policy": str(runtime.threshold_policy),
    }
    report["corpus"] = {
        "schema_version": corpus.schema_version,
        "case_count": len(corpus.cases),
        "source_features": len(corpus.source_features),
        "score_absolute_tolerance": corpus.score_absolute_tolerance,
        "roles": _role_counts(corpus),
    }
    report["coverage"] = check_coverage(corpus, engine)
    return runtime, engine, corpus


def _stable_code(error: BaseException) -> str:
    """把底层异常折叠成稳定分类码：优先取它自己的 ``code``，否则退回类别名。"""
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    return type(error).__name__


def _role_counts(corpus: RegressionCorpus) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in corpus.cases:
        counts[case.role] = counts.get(case.role, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


# --- 报告组装 -----------------------------------------------------------


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_version": __version__,
        **_optional_versions(("lightgbm", "joblib", "sklearn", "pandas", "numpy")),
    }


def _optional_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover - 依赖缺失会在加载阶段先暴露
            versions[name] = "unavailable"
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _regression_summary(outcome: RegressionOutcome) -> dict[str, Any]:
    return {
        "cases_total": outcome.cases_total,
        "cases_passed": outcome.cases_total - outcome.cases_failed,
        "cases_failed": outcome.cases_failed,
        "failed_case_ids": [failure.case_id for failure in outcome.failures],
        "failed_fields": failed_fields_by_case(outcome),
        "max_score_absolute_deviation": outcome.max_score_absolute_deviation,
        "deterministic": outcome.deterministic,
        "expected_distribution": outcome.expected_distribution,
        "actual_distribution": outcome.actual_distribution,
        "distribution_matches_expected": outcome.distribution_matches,
    }


def _pytest_summary(outcome: PytestOutcome, config: QualityGateConfig) -> dict[str, Any]:
    return {
        "status": outcome.status,
        "command": {
            "program": PYTEST_PROGRAM_PLACEHOLDER,
            "arguments": list(config.pytest_arguments),
            "cwd": PYTEST_CWD_PLACEHOLDER,
            "shell": False,
            "timeout_seconds": config.pytest_timeout_seconds,
        },
        "exit_code": outcome.exit_code,
        "duration_seconds": outcome.duration_seconds,
        "child_reaped": outcome.child_reaped,
        "counts": {
            "passed": outcome.passed,
            "skipped": outcome.skipped,
            "failed": outcome.failed,
            "errors": outcome.errors,
        },
    }


# --- 主流程 -------------------------------------------------------------


def run_gate(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    regression_only: bool = False,
    pytest_runner: Callable[..., PytestOutcome] = run_pytest,
    fallback_report_path: str = DEFAULT_REPORT_PATH,
) -> GateOutcome:
    """执行一次门禁。

    Args:
        config_path: 项目相对的配置路径。
        regression_only: 只跑 preflight + regression + report（专项测试/诊断用）。
        pytest_runner: pytest 阶段的可注入接缝；默认 :func:`run_pytest`。
        fallback_report_path: 配置本身无法解析时使用的报告路径（默认即冻结路径）。

    Returns:
        :class:`GateOutcome`。报告写失败时 ``report`` 与 ``report_written`` 均为 ``None``/``False``，
        退出码为 6。
    """
    started = time.perf_counter()
    report: dict[str, Any] = {
        "work_package": WORK_PACKAGE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "overall_status": "failed",
        "exit_code": None,
        "error_code": None,
        "stages": [],
        "inputs": {},
        "versions": {},
        "corpus": {},
        "coverage": {},
        "regression": {},
        "pytest": {"status": "not_run"},
        "environment": _environment(),
        "limitations": list(LIMITATIONS),
    }
    stages: list[dict[str, Any]] = []
    allowed_digests: set[str] = set()
    forbidden_tokens: frozenset[str] = frozenset()
    report_path = (PROJECT_ROOT / fallback_report_path).resolve()

    exit_code = EXIT_OK
    error_code: str | None = None
    config: QualityGateConfig | None = None
    runtime = engine = corpus = None

    # --- 阶段 1：预检 ---
    try:
        config = QualityGateConfig.from_yaml(config_path)
        report_path = config.report_target()
        allowed_digests.update(
            frozen.sha256
            for frozen in (
                config.service_config,
                config.bundle,
                config.policy,
                config.corpus,
            )
        )
        runtime, engine, corpus = preflight(config, report)
        forbidden_tokens = forbidden_report_tokens(corpus)
        stages.append({"name": "preflight", "status": "passed", "error_code": None})
    except QualityGateError as error:
        exit_code, error_code = EXIT_PREFLIGHT, error.code
        stages.append({"name": "preflight", "status": "failed", "error_code": error.code})
    except Exception:  # noqa: BLE001 - 未预期内部错误：失败关闭，只记类别
        exit_code, error_code = EXIT_PREFLIGHT, QualityGateErrorCode.INTERNAL_ERROR
        stages.append(
            {"name": "preflight", "status": "failed", "error_code": QualityGateErrorCode.INTERNAL_ERROR}
        )

    # 无论预检成功与否，都把"本次实际算出的"摘要加入脱敏允许清单：指纹不符时失败报告里
    # 会写出**不匹配的实际摘要**，若不放行，报告会被自己的脱敏断言拒绝并错误地返回 6。
    allowed_digests.update(
        str(item["actual_sha256"])
        for item in report["inputs"].values()
        if item.get("actual_sha256")
    )

    # --- 阶段 2：回归 ---
    if exit_code == EXIT_OK and config is not None and corpus is not None:
        try:
            outcome = run_regression(runtime, engine, corpus)
            report["regression"] = _regression_summary(outcome)
            residual = aggregate_error_code(outcome)
            if residual is None:
                stages.append({"name": "regression", "status": "passed", "error_code": None})
            else:
                exit_code, error_code = EXIT_REGRESSION, residual
                stages.append({"name": "regression", "status": "failed", "error_code": residual})
        except QualityGateError as error:
            exit_code, error_code = EXIT_REGRESSION, error.code
            stages.append({"name": "regression", "status": "failed", "error_code": error.code})
        except Exception:  # noqa: BLE001 - 同上
            exit_code, error_code = EXIT_REGRESSION, QualityGateErrorCode.INTERNAL_ERROR
            stages.append(
                {"name": "regression", "status": "failed", "error_code": QualityGateErrorCode.INTERNAL_ERROR}
            )

    # --- 阶段 3：pytest ---
    if exit_code == EXIT_OK and config is not None:
        if regression_only:
            report["pytest"] = {"status": "skipped_by_regression_only"}
            stages.append({"name": "pytest", "status": "skipped", "error_code": None})
        else:
            try:
                outcome = pytest_runner(
                    build_pytest_command(config),
                    cwd=PROJECT_ROOT,
                    timeout_seconds=config.pytest_timeout_seconds,
                )
                report["pytest"] = _pytest_summary(outcome, config)
                stage_failure = _pytest_failure(outcome)
                if stage_failure is None:
                    stages.append({"name": "pytest", "status": "passed", "error_code": None})
                else:
                    code, exit_code = stage_failure
                    error_code = code
                    stages.append({"name": "pytest", "status": "failed", "error_code": code})
            except Exception:  # noqa: BLE001 - 注入实现异常同样失败关闭
                exit_code, error_code = EXIT_PYTEST, QualityGateErrorCode.INTERNAL_ERROR
                stages.append(
                    {"name": "pytest", "status": "failed", "error_code": QualityGateErrorCode.INTERNAL_ERROR}
                )

    report["stages"] = stages
    report["exit_code"] = exit_code
    report["error_code"] = error_code
    report["overall_status"] = "passed" if exit_code == EXIT_OK else "failed"
    report["duration_seconds"] = round(time.perf_counter() - started, 3)

    try:
        write_report(
            report_path,
            report,
            allowed_digests=allowed_digests,
            forbidden_tokens=forbidden_tokens,
        )
    except Exception:  # noqa: BLE001 - 写入失败（含脱敏断言不通过）一律退出 6，不外泄 traceback
        return GateOutcome(
            exit_code=EXIT_REPORT,
            report_path=_relative_report_path(report_path),
            report=None,
            report_written=False,
        )

    return GateOutcome(
        exit_code=exit_code,
        report_path=_relative_report_path(report_path),
        report=report,
        report_written=True,
    )


def _relative_report_path(path: Path) -> str:
    """把报告路径写成项目相对形式；越界时退回冻结的默认常量（绝不写绝对路径）。"""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:  # pragma: no cover - 越界会在配置阶段先被拒绝
        return DEFAULT_REPORT_PATH


def _pytest_failure(outcome: PytestOutcome) -> tuple[str, int] | None:
    """把 pytest 结果映射成 ``(error_code, exit_code)``；通过时返回 ``None``。"""
    if outcome.status == "passed":
        return None
    if outcome.status == "timeout":
        return QualityGateErrorCode.PYTEST_TIMEOUT, EXIT_PYTEST_TIMEOUT
    if outcome.status == "spawn_failed":
        return QualityGateErrorCode.PYTEST_SPAWN_FAILED, EXIT_PYTEST
    return QualityGateErrorCode.PYTEST_FAILED, EXIT_PYTEST


# --- 命令行 -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CW-009 一键质量门禁：产物预检 → 固定回归 → 完整 pytest → 安全报告。"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"门禁配置（项目相对路径），默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--regression-only",
        action="store_true",
        help="只跑 preflight + regression + report，不运行 pytest（供专项测试与诊断）。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。返回冻结退出码（0/2/3/4/5/6）。"""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    outcome = run_gate(args.config, regression_only=bool(args.regression_only))
    _print_summary(args.config, outcome)
    return outcome.exit_code


def _print_summary(config_path: str, outcome: GateOutcome) -> None:
    """控制台摘要：只输出相对路径、计数与稳定错误码，不输出任何取值或绝对路径。"""
    stream = sys.stdout if outcome.exit_code == EXIT_OK else sys.stderr
    report = outcome.report
    print(f"配置：{config_path}", file=stream)
    if report is None:
        print(f"质量门禁失败：报告未能写入（退出码 {EXIT_REPORT}）", file=stream)
        return
    for stage in report.get("stages", []):
        suffix = f"（{stage['error_code']}）" if stage.get("error_code") else ""
        print(f"  {stage['name']:<10} {stage['status']}{suffix}", file=stream)
    regression = report.get("regression") or {}
    if regression:
        print(
            f"  回归：{regression.get('cases_passed', 0)}/{regression.get('cases_total', 0)} 通过，"
            f"最大分数绝对偏差 {regression.get('max_score_absolute_deviation')}",
            file=stream,
        )
    counts = (report.get("pytest") or {}).get("counts")
    if counts:
        print(
            f"  pytest：{counts.get('passed', 0)} passed / {counts.get('skipped', 0)} skipped / "
            f"{counts.get('failed', 0)} failed / {counts.get('errors', 0)} errors",
            file=stream,
        )
    if outcome.exit_code == EXIT_OK:
        print(f"质量门禁通过（退出码 0）；报告：{outcome.report_path}", file=stream)
    else:
        print(
            f"质量门禁失败：退出码 {outcome.exit_code}，错误码 {report.get('error_code')}",
            file=stream,
        )
    print(
        "提醒：固定回归集是 golden master，不是模型效果、公平性或生产就绪性的证据。",
        file=stream,
    )


def load_report(path: str | Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    """读取报告（仅供测试与人工诊断）。"""
    return json.loads(resolve_within_project(str(path), "report").read_text(encoding="utf-8"))


__all__ = [
    "EXIT_OK",
    "EXIT_PREFLIGHT",
    "EXIT_REGRESSION",
    "EXIT_PYTEST",
    "EXIT_PYTEST_TIMEOUT",
    "EXIT_REPORT",
    "GateOutcome",
    "PytestOutcome",
    "build_parser",
    "build_pytest_command",
    "load_report",
    "main",
    "preflight",
    "run_gate",
    "run_pytest",
]
