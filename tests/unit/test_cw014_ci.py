"""CW-014 CI 就绪的区分性测试：工作流边界、依赖约束、输入供给与提交卫生。

这些用例守护的是**能被审查的性质**，不是"YAML 存在就算过"：

- 工作流只由受信任触发（`workflow_dispatch` / 默认分支 push），**没有** `pull_request`
  或 `pull_request_target`——外部 PR 拿不到 token，也读不到模型资产；
- 权限最小（只有 `contents: read`，没有任何 `write`）；第三方 action 都带版本；
- 输入校验步骤**先于**总门禁步骤，且缺输入/摘要不符会失败关闭；
- 依赖约束文件是精确锁定（`==`），包含数值关键包；
- `configs/ci_inputs.yaml` 的 bundle 摘要是对门禁配置的**转述**，不允许漂移；
- `.gitignore` 排除本机协作记录与敏感类别，但不排除 CI 自身的文件。

输入供给的失败路径用**暂存根**演练：真实 `configs/`、真实 bundle 与真实默认库都不参与。
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

import yaml

from ai_detector.config import PROJECT_ROOT
from scripts import ci_fetch_inputs as fetch

WORKFLOW_REL = ".github/workflows/quality-gate.yml"
CONSTRAINTS_REL = "constraints/cw014-ci.txt"
CI_INPUTS_REL = "configs/ci_inputs.yaml"
GATE_CONFIG_REL = "configs/quality_gate.yaml"
FIXTURE_REL = "tests/fixtures/cw014/rule_baseline_state.json"

STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests" / "cw014"

PAYLOAD = b"cw014-synthetic-release-asset-" + b"r" * 48


def _digest(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# --- 工作流审计 ----------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load((PROJECT_ROOT / WORKFLOW_REL).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    """PyYAML 按 YAML 1.1 把裸 `on:` 解析成布尔 True，两种写法都要认。"""
    return workflow.get("on") or workflow.get(True) or {}


def test_workflow_is_triggered_only_by_trusted_events(workflow: dict) -> None:
    triggers = _triggers(workflow)
    assert "workflow_dispatch" in triggers
    assert "push" in triggers
    assert "pull_request" not in triggers
    assert "pull_request_target" not in triggers, "PR 触发会让外部代码拿到 token 与模型资产"


def test_workflow_permissions_are_minimal(workflow: dict) -> None:
    permissions = workflow["permissions"]
    assert permissions == {"contents": "read"}, permissions
    text = (PROJECT_ROOT / WORKFLOW_REL).read_text(encoding="utf-8")
    assert "write" not in text, "不得出现任何写权限"


def test_workflow_runs_on_a_pinned_runner_and_python(workflow: dict) -> None:
    job = workflow["jobs"]["quality-gate"]
    assert job["runs-on"] == "windows-latest", "固定与本机已验收基线同平台"
    versions = [
        step["with"]["python-version"]
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/setup-python")
    ]
    assert versions == ["3.14"], versions


def test_every_third_party_action_is_version_pinned(workflow: dict) -> None:
    for step in workflow["jobs"]["quality-gate"]["steps"]:
        uses = step.get("uses")
        if uses:
            assert "@" in uses, uses
            version = uses.split("@", 1)[1]
            assert version and version != "main", uses


def test_only_the_automatic_token_is_referenced(workflow: dict) -> None:
    text = (PROJECT_ROOT / WORKFLOW_REL).read_text(encoding="utf-8")
    assert "secrets.GITHUB_TOKEN" in text
    assert "secrets." in text and text.count("secrets.") == text.count("secrets.GITHUB_TOKEN"), (
        "除自动提供的 GITHUB_TOKEN 外不得引用任何 secret"
    )


def test_inputs_are_verified_before_the_gate_runs(workflow: dict) -> None:
    runs = [
        step["run"]
        for step in workflow["jobs"]["quality-gate"]["steps"]
        if "run" in step
    ]
    joined = "\n".join(runs)
    assert "scripts/ci_fetch_inputs.py" in joined, "必须先校验输入"
    assert "scripts/run_project_gate.py" in joined, "必须原样运行本地总门禁"
    fetch_at = next(i for i, run in enumerate(runs) if "ci_fetch_inputs.py" in run)
    gate_at = next(i for i, run in enumerate(runs) if "run_project_gate.py" in run)
    assert fetch_at < gate_at, "输入校验必须发生在总门禁之前"
    assert "joblib.load" not in joined


def test_workflow_installs_from_the_constraint_file(workflow: dict) -> None:
    joined = "\n".join(
        step["run"] for step in workflow["jobs"]["quality-gate"]["steps"] if "run" in step
    )
    assert f"-c {CONSTRAINTS_REL}" in joined, "依赖必须受约束文件锁定"
    assert "site-packages" not in joined, "不得复制或引用开发机 site-packages"


def test_full_pytest_is_run_only_through_the_gate(workflow: dict) -> None:
    """工作流里不得另起一次全量 pytest（只允许总门禁内部那一次）。"""
    joined = "\n".join(
        step["run"] for step in workflow["jobs"]["quality-gate"]["steps"] if "run" in step
    )
    assert "pytest" not in joined.replace("run_project_gate.py", ""), joined


# --- 依赖约束 ------------------------------------------------------------


def test_constraints_pin_every_package_exactly() -> None:
    lines = [
        line.strip()
        for line in (PROJECT_ROOT / CONSTRAINTS_REL).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines, "约束文件不得为空"
    for line in lines:
        name, _, version = line.partition("==")
        assert name and version, f"必须精确锁定：{line}"
        assert not any(op in version for op in (">", "<", "*", ",")), line
    joined = "\n".join(lines)
    for critical in ("numpy==", "pandas==", "scikit-learn==", "lightgbm==", "joblib==", "ruff=="):
        assert critical in joined, critical


def test_constraints_document_the_platform_and_tolerance_risk() -> None:
    text = (PROJECT_ROOT / CONSTRAINTS_REL).read_text(encoding="utf-8")
    assert "1e-12" in text and "windows-latest" in text, "必须写明跨平台数值风险"


# --- 输入清单与门禁配置一致 ---------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load((PROJECT_ROOT / CI_INPUTS_REL).read_text(encoding="utf-8"))["ci_inputs"]


def test_manifest_bundle_entry_is_authoritative_transcription(manifest: dict) -> None:
    """清单里的 bundle 必须与门禁配置逐项一致——它是转述，不是第二授权来源。"""
    gate = yaml.safe_load((PROJECT_ROOT / GATE_CONFIG_REL).read_text(encoding="utf-8"))[
        "quality_gate"
    ]["inputs"]["bundle"]
    entry = manifest["inputs"]["bundle"]
    assert entry["path"] == gate["path"]
    assert entry["sha256"] == gate["sha256"]


def test_manifest_declares_a_known_supply_for_every_input(manifest: dict) -> None:
    entries = manifest["inputs"]
    assert set(entries) == {"bundle", "rule_baseline_state", "default_decisions_db"}
    for label, entry in entries.items():
        assert entry["supply"] in fetch.KNOWN_SUPPLY, label
        assert len(entry["sha256"]) == 64, label
        assert isinstance(entry["size"], int) and entry["size"] > 0, label
    assert entries["bundle"]["supply"] == fetch.SUPPLY_RELEASE
    assert entries["rule_baseline_state"]["supply"] == fetch.SUPPLY_FIXTURE
    assert entries["default_decisions_db"]["supply"] == fetch.SUPPLY_NOT_REQUIRED


def test_committed_fixture_is_byte_identical_to_the_expected_digest(manifest: dict) -> None:
    entry = manifest["inputs"]["rule_baseline_state"]
    fixture = PROJECT_ROOT / entry["fixture"]
    assert fixture.is_file()
    assert _digest(fixture.read_bytes()) == entry["sha256"]
    assert fixture.stat().st_size == entry["size"]
    assert FIXTURE_REL == entry["fixture"]


# --- 输入供给的失败关闭（用暂存根演练） ---------------------------------


class Staged:
    """一个隔离的"项目根"：只含暂存配置与被供给的文件。

    清单里的路径是**相对这个暂存根**的（`provide_inputs(root)` 的约定就是如此），
    因此演练完全不碰真实的 `configs/`、真实 bundle 与真实默认库。
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.bundle_rel = "model/bundle.joblib"
        self.state_rel = "model/state.json"
        self.fixture_rel = "fixture/state.json"
        self.db_rel = "runtime/decisions.sqlite3"
        self.payload = PAYLOAD
        (directory / "configs").mkdir(parents=True, exist_ok=True)
        (directory / "fixture").mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        bundle_supply: str = fetch.SUPPLY_RELEASE,
        db_supply: str | None = None,
        bundle_digest: str | None = None,
        gate_digest: str | None = None,
    ) -> None:
        digest = bundle_digest or _digest(self.payload)
        ci = {
            "ci_inputs": {
                "schema_version": "cw014_ci_inputs_v1",
                "inputs": {
                    "bundle": {
                        "path": self.bundle_rel, "sha256": digest, "size": len(self.payload),
                        "supply": bundle_supply, "fixture": None,
                    },
                    "rule_baseline_state": {
                        "path": self.state_rel,
                        "sha256": _digest(self.fixture_bytes()), "size": len(self.fixture_bytes()),
                        "supply": fetch.SUPPLY_FIXTURE, "fixture": self.fixture_rel,
                    },
                    "default_decisions_db": {
                        "path": self.db_rel, "sha256": "0" * 64, "size": 1,
                        "supply": db_supply or fetch.SUPPLY_NOT_REQUIRED, "fixture": None,
                    },
                },
            }
        }
        gate = {
            "quality_gate": {
                "inputs": {"bundle": {"path": self.bundle_rel, "sha256": gate_digest or digest}}
            }
        }
        (self.directory / "configs/ci_inputs.yaml").write_text(
            yaml.safe_dump(ci, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (self.directory / "configs/quality_gate.yaml").write_text(
            yaml.safe_dump(gate, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def fixture_bytes(self) -> bytes:
        return b'{"name":"staged","thresholds":{"x":1.0}}'

    def write_fixture(self, data: bytes | None = None) -> None:
        path = self.directory / self.fixture_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if data is not None else self.fixture_bytes())

    def write_bundle(self, data: bytes | None = None) -> None:
        path = self.directory / self.bundle_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if data is not None else self.payload)


@pytest.fixture
def staged() -> Staged:
    directory = STAGE_ROOT / uuid.uuid4().hex
    staged = Staged(directory)
    staged.write()
    try:
        yield staged
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _specs(staged: Staged) -> list[fetch.Spec]:
    return fetch.load_specs(staged.directory)


def test_release_asset_must_be_present_and_match(staged: Staged) -> None:
    with pytest.raises(fetch.InputError) as caught:
        fetch.provide_inputs(staged.directory)
    assert caught.value.code == "input_missing"
    assert caught.value.exit_code == fetch.EXIT_MISSING_OR_MISMATCH


def test_release_asset_with_wrong_bytes_fails_closed(staged: Staged) -> None:
    staged.write_bundle(b"a different, tampered asset" + b"t" * 32)
    with pytest.raises(fetch.InputError) as caught:
        fetch.provide_inputs(staged.directory)
    assert caught.value.code == "input_digest_mismatch"
    assert caught.value.exit_code == fetch.EXIT_MISSING_OR_MISMATCH


def test_committed_fixture_is_placed_and_is_idempotent(staged: Staged) -> None:
    staged.write_fixture()
    staged.write_bundle()
    first = fetch.provide_inputs(staged.directory)
    placed = staged.directory / staged.state_rel
    assert first["inputs"]["rule_baseline_state"]["status"] == "placed"
    assert placed.read_bytes() == staged.fixture_bytes()

    before = placed.stat().st_mtime_ns
    second = fetch.provide_inputs(staged.directory)
    assert second["inputs"]["rule_baseline_state"]["status"] == "already_present"
    assert placed.stat().st_mtime_ns == before, "已一致时不得重写"


def test_tampered_fixture_is_rejected_and_nothing_is_placed(staged: Staged) -> None:
    staged.write_fixture(b"tampered fixture bytes")
    staged.write_bundle()
    with pytest.raises(fetch.InputError) as caught:
        fetch.provide_inputs(staged.directory)
    assert caught.value.code == "input_digest_mismatch"
    assert not (staged.directory / staged.state_rel).exists()


def test_not_required_input_absent_is_fine(staged: Staged) -> None:
    staged.write_fixture()
    staged.write_bundle()
    report = fetch.provide_inputs(staged.directory)
    assert report["inputs"]["default_decisions_db"]["status"] == "not_required"


def test_not_required_input_present_but_wrong_fails_closed(staged: Staged) -> None:
    staged.write_fixture()
    staged.write_bundle()
    db = staged.directory / staged.db_rel
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"a foreign database")
    with pytest.raises(fetch.InputError) as caught:
        fetch.provide_inputs(staged.directory)
    assert caught.value.code == "input_digest_mismatch"


def test_manifest_that_contradicts_the_gate_config_fails_closed(staged: Staged) -> None:
    """清单不是第二授权来源：与门禁配置不一致时以退出码 3 失败关闭。"""
    staged.write(bundle_digest="a" * 64, gate_digest=_digest(PAYLOAD))  # 清单写 A，门禁配置写真实摘要
    with pytest.raises(fetch.InputError) as caught:
        fetch.load_specs(staged.directory)
    assert caught.value.code == "bundle_not_authoritative"
    assert caught.value.exit_code == fetch.EXIT_MANIFEST


def test_unknown_supply_kind_is_rejected(staged: Staged) -> None:
    staged.write(bundle_supply="ftp")
    with pytest.raises(fetch.InputError) as caught:
        fetch.load_specs(staged.directory)
    assert caught.value.code == "supply_unknown"


def test_report_and_console_carry_no_absolute_paths(staged: Staged, capsys) -> None:
    staged.write_fixture()
    staged.write_bundle()
    report = fetch.provide_inputs(staged.directory)
    rendered = json.dumps(report, ensure_ascii=False)
    assert str(PROJECT_ROOT) not in rendered
    for entry in report["inputs"].values():
        assert not Path(entry["path"]).is_absolute(), entry


def test_spec_loading_reports_every_declared_input(staged: Staged) -> None:
    labels = {spec.label for spec in _specs(staged)}
    assert labels == {"bundle", "rule_baseline_state", "default_decisions_db"}


# --- 提交卫生 ------------------------------------------------------------


@pytest.fixture(scope="module")
def gitignore() -> list[str]:
    return [
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.mark.parametrize(
    "pattern", [".handoff/", "WORK_LOG.md", "CLAUDE.md", "artifacts/**/*", ".venv/", "data/raw/**/*.csv"]
)
def test_gitignore_excludes_local_records_and_sensitive_inputs(
    gitignore: list[str], pattern: str
) -> None:
    assert pattern in gitignore, pattern


@pytest.mark.parametrize(
    "pattern", [".github/workflows/quality-gate.yml", "constraints/cw014-ci.txt",
                "configs/ci_inputs.yaml", "tests/fixtures/cw014/rule_baseline_state.json"]
)
def test_ci_files_are_not_ignored(gitignore: list[str], pattern: str) -> None:
    """CI 自身需要的文件必须能被跟踪（不能被上面的排除规则误伤）。"""
    import fnmatch

    for rule in gitignore:
        if rule.startswith("!"):
            continue
        assert not fnmatch.fnmatch(pattern, rule), f"{pattern} 被 {rule} 排除"


def test_ci_inputs_are_declared_but_not_secrets() -> None:
    text = (PROJECT_ROOT / CI_INPUTS_REL).read_text(encoding="utf-8")
    assert "f02bb50e" in text
    for forbidden in ("password", "BEGIN PRIVATE KEY", "api_key"):
        assert forbidden not in text.lower(), forbidden
