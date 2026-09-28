"""CW-012 CI 输入审计：在**临时副本**里证明干净检出的 bundle 缺口会失败关闭。

## 为什么不是"把整个项目复制到 tmp"

本项目以 **editable 方式**安装在 ``.venv`` 里（``__editable__.ai_detector-0.1.0.pth``
注册的是 MetaPathFinder，优先级高于 ``sys.path``）。因此无论 cwd 与 ``sys.path`` 怎么设，
``import ai_detector`` 都会解析到**真实仓库**、``PROJECT_ROOT`` 也永远是真实项目根。
在 tmp 里复制一份 ``src/`` 并不能改变这一点——实测确认过，所以本模块改用**暂存配置**：
把 CW-009 门禁配置复制到项目内被 Git 忽略的 ``artifacts/tmp_tests/cw012/`` 下，
只改 ``bundle.path`` 与 ``report.path``，指向该暂存目录。

这样既能构造"bundle 缺失 / 字节不符"的真实反例，又满足两条硬约束：

- **不碰真实 bundle、不碰默认库、不覆盖真实 CW-009 报告**（未尾用例把它钉成断言）；
- 预检失败时**不会**去加载模型，因此不需要真的有一份可加载的 bundle。

被反复验证的三件事：

1. bundle 缺失或字节不符 → CW-009 预检以退出码 2 失败关闭，且 ``actual_sha256`` 分别
   是 ``null`` 与真实摘要（不伪造匹配）；
2. 在同一个暂存配置下跑总门禁：静态阶段如实通过，但质量门禁阶段失败 →
   总门禁退出 5、``status=failed``，**静态阶段不会把整体刷成绿**；
3. 每次运行后真实仓库产物逐字节未变。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import yaml

from ai_detector.config import PROJECT_ROOT
from project_gate import ci_inputs, gate

GATE_CONFIG_REL = "configs/quality_gate.yaml"

#: 暂存目录：项目内、被 ``.gitignore`` 的 ``artifacts/**`` 规则排除。
STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests" / "cw012"

#: 真实产物：本模块必须**逐字节**不动它们。
REAL_REPORT = PROJECT_ROOT / gate.QUALITY_GATE_REPORT_REL
REAL_BUNDLE = PROJECT_ROOT / "artifacts/models/cw004/lightgbm_bundle.joblib"


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


class StagedGate:
    """一份指向暂存 bundle / 暂存报告的 CW-009 配置。"""

    def __init__(self, directory: Path, config_rel: str, report_path: Path, bundle_path: Path):
        self.directory = directory
        self.config_rel = config_rel
        self.report_path = report_path
        self.bundle_path = bundle_path

    def report(self) -> dict:
        return json.loads(self.report_path.read_text(encoding="utf-8"))


@pytest.fixture
def staged_gate() -> StagedGate:
    """把门禁配置暂存到项目内被忽略的目录；退出时清理。"""
    directory = STAGE_ROOT / uuid.uuid4().hex
    directory.mkdir(parents=True)
    relative = directory.relative_to(PROJECT_ROOT).as_posix()
    raw = yaml.safe_load((PROJECT_ROOT / GATE_CONFIG_REL).read_text(encoding="utf-8"))
    raw["quality_gate"]["report"]["path"] = f"{relative}/report.json"
    raw["quality_gate"]["inputs"]["bundle"]["path"] = f"{relative}/bundle.joblib"
    config_path = directory / "quality_gate.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    staged = StagedGate(
        directory=directory,
        config_rel=config_path.relative_to(PROJECT_ROOT).as_posix(),
        report_path=directory / "report.json",
        bundle_path=directory / "bundle.joblib",
    )
    try:
        yield staged
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _run_gate_command(staged: StagedGate) -> subprocess.CompletedProcess:
    """按官方命令行方式运行 CW-009 门禁（不使用 shell，cwd 固定在项目根）。"""
    return subprocess.run(
        [
            sys.executable,
            gate.QUALITY_GATE_SCRIPT_REL,
            "--config",
            staged.config_rel,
        ],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )


# --- 输入清单与门禁配置一致 ----------------------------------------------


def test_inventory_covers_every_input_the_gate_config_freezes() -> None:
    raw = yaml.safe_load((PROJECT_ROOT / GATE_CONFIG_REL).read_text(encoding="utf-8"))
    declared = set(raw["quality_gate"]["inputs"])
    registered = {entry.label for entry in ci_inputs.GATE_INPUTS}
    assert declared == registered, "门禁配置里的输入必须逐项登记在 CI 输入清单里"


def test_inventory_paths_and_digests_match_the_gate_config() -> None:
    """清单里的路径与冻结摘要必须与门禁配置一致（防止清单写错文件）。"""
    raw = yaml.safe_load((PROJECT_ROOT / GATE_CONFIG_REL).read_text(encoding="utf-8"))
    for label, section in raw["quality_gate"]["inputs"].items():
        entry = ci_inputs.by_label(label)
        assert entry.path == section["path"], label
        assert section["sha256"] in entry.verification, f"{label} 的清单说明必须写明冻结摘要"


def test_the_bundle_is_the_only_ignored_gate_input() -> None:
    """bundle 是唯一被 `.gitignore` 排除的门禁输入——这正是 CI 缺口的所在。"""
    ignored = {entry.label for entry in ci_inputs.ignored_inputs()}
    assert ignored == {"bundle"}
    bundle = ci_inputs.by_label("bundle")
    assert not bundle.committed
    assert "没有可信供给方式" in bundle.trust
    assert bundle.missing_behaviour == ci_inputs.FAIL_CLOSED


def test_gitignore_excludes_the_bundle_but_keeps_the_readme() -> None:
    patterns = [
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "artifacts/**/*" in patterns
    assert "!artifacts/README.md" in patterns, "只放行说明文件，不放行模型二进制"


def test_every_committed_input_exists_in_this_working_copy() -> None:
    """本工作副本里已提交输入都在——这解释了为什么本地能跑通而干净检出不能。"""
    for entry in ci_inputs.CI_INPUTS:
        if entry.committed and not entry.path.startswith("（"):
            assert (PROJECT_ROOT / entry.path).exists(), entry.path


def test_static_scope_paths_all_exist() -> None:
    """静态检查范围里的每个路径都必须真实存在，否则 ruff 会报 E902。"""
    from project_gate.static_checks import STATIC_PATHS

    for relative in STATIC_PATHS:
        assert (PROJECT_ROOT / relative).exists(), relative


def test_ci_readiness_document_covers_every_registered_input() -> None:
    """就绪清单必须写到每个登记输入，且明确未解决的两项前置条件。"""
    text = (PROJECT_ROOT / "docs/ci_readiness.md").read_text(encoding="utf-8")
    for entry in ci_inputs.CI_INPUTS:
        # 占位路径（如 pip extra）不是文件，按名字核对；其余按真实路径核对。
        token = entry.path if not entry.path.startswith("（") else entry.label
        if entry.label == "quality_extra":
            token = "pyproject.toml"
        assert token in text or entry.label in text, entry.label
    for extra in ("static", "service", "tree", "test"):
        assert extra in text, extra
    for required in ("input_missing", "input_hash_mismatch", "未确定", "未完成"):
        assert required in text, required
    assert ".github" in text, "必须说明本包不创建任何平台的流水线文件"


# --- 真实门禁：bundle 缺失 / 字节不符 → 预检失败关闭 ---------------------


def test_missing_bundle_fails_preflight_and_is_reported_as_null(staged_gate: StagedGate) -> None:
    assert not staged_gate.bundle_path.exists()
    result = _run_gate_command(staged_gate)

    assert result.returncode == 2, result.stdout + result.stderr
    report = staged_gate.report()
    assert report["overall_status"] == "failed"
    assert report["error_code"] == "input_missing", "文件不存在与字节不符是两类失败"
    bundle = report["inputs"]["bundle"]
    assert bundle["match"] is False
    assert bundle["actual_sha256"] is None, "缺失必须是 null，不能伪造成匹配"
    for label in ("service_config", "policy", "corpus"):
        assert report["inputs"][label]["match"] is True, label
    assert report["pytest"] == {"status": "not_run"}, "预检失败不得运行 pytest"
    assert not report["regression"], "预检失败不得运行回归"


def test_tampered_bundle_fails_preflight_with_the_actual_digest(
    staged_gate: StagedGate,
) -> None:
    """bundle 存在但字节不符（例如来源不明的替代品）同样失败关闭。"""
    payload = bytearray(REAL_BUNDLE.read_bytes())
    payload[0] ^= 0xFF
    staged_gate.bundle_path.write_bytes(bytes(payload))

    result = _run_gate_command(staged_gate)
    assert result.returncode == 2, result.stdout + result.stderr
    report = staged_gate.report()
    assert report["error_code"] == "input_hash_mismatch"
    bundle = report["inputs"]["bundle"]
    assert bundle["match"] is False
    assert bundle["actual_sha256"] is not None, "字节不符时要列出实际摘要以便定位"
    assert bundle["actual_sha256"] != bundle["declared_sha256"]


# --- 总门禁：静态通过但质量门禁失败 → 整体失败 --------------------------


def test_static_stage_does_not_fake_success_when_the_gate_fails(
    staged_gate: StagedGate,
) -> None:
    """总门禁在缺 bundle 的暂存配置下：静态通过、质量门禁失败 → 整体失败、退出 5。"""
    outcome = gate.run_project_gate(config_rel=staged_gate.config_rel)

    assert outcome.status == gate.STATUS_FAILED
    assert outcome.exit_code == gate.EXIT_QUALITY_GATE
    assert outcome.error_code == "quality_gate_failed"
    assert outcome.static is not None and outcome.static.passed, "静态阶段应如实通过"

    report = gate.build_report(outcome)
    quality = report["phases"][gate.PHASE_QUALITY_GATE]
    assert quality["status"] == "failed"
    assert quality["exit_code"] == 2
    assert quality["cw009_status"] == "failed"
    assert quality["inputs_match"]["bundle"] is False
    assert quality["inputs_match"]["service_config"] is True
    # 整体结论必须是失败——静态阶段没有把结果刷成绿。
    assert report["status"] == "failed" and report["exit_code"] != 0


def test_total_gate_report_contains_no_absolute_paths(staged_gate: StagedGate) -> None:
    outcome = gate.run_project_gate(config_rel=staged_gate.config_rel)
    rendered = json.dumps(gate.build_report(outcome), ensure_ascii=False)
    assert str(PROJECT_ROOT) not in rendered
    assert str(staged_gate.directory) not in rendered
    assert "::" not in rendered, "不得含 pytest 节点 ID"
    assert "stdout" not in rendered and "stderr" not in rendered


# --- 反例必须无副作用：真实产物逐字节不变 --------------------------------


def test_audit_never_touches_real_artifacts(staged_gate: StagedGate) -> None:
    """审计前后：真实 CW-009 报告与真实 bundle 逐字节未变。"""
    before = (_digest(REAL_REPORT), _digest(REAL_BUNDLE))
    _run_gate_command(staged_gate)
    gate.run_project_gate(config_rel=staged_gate.config_rel)
    after = (_digest(REAL_REPORT), _digest(REAL_BUNDLE))
    assert before == after, "CI 输入审计不得改动真实报告或真实 bundle"
    assert before[0] is not None and before[1] is not None
