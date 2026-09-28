"""CW-013 隔离源码副本验证：解析必须指向副本，缺 bundle 必须失败关闭。

## 为什么这里只跑 ``--regression-only``

"导入后跑**原样**总门禁"的完整运行（含全量 pytest）是**记录在案的证据**，由
``scripts/verify_isolated_bundle_gate.py --full`` 执行（真实命令与计数见 `result.md`）。
本模块刻意**只**跑 CW-009 的 ``--regression-only``：

- 缺 bundle 的失败发生在 **preflight**，早于 pytest，因此 ``--regression-only``
  给出**完全相同**的失败证据；
- 若在这里跑完整总门禁，它会运行整个测试套件，而套件里又有本模块 →
  每一层都会再生成下一层，构成无界递归。CW-009 的集成测试出于同样理由只用
  ``--regression-only``。

本模块覆盖的三件事：

1. **副本不带运行产物**：没有 ``artifacts/**``，但已提交源码齐全
   （覆盖 ``STATIC_PATHS`` 的每一项，否则 ruff 会报 E902 把"副本不完整"误报成违规）。
2. **解析不回指真实仓库**：副本里独立 venv 的 ``ai_detector.__file__`` 与
   ``PROJECT_ROOT`` 都落在副本内。CW-012 踩过的坑——真实 venv 的 editable 安装是
   MetaPathFinder，优先级高于 ``sys.path``，所以只设 ``PYTHONPATH`` 不够。
3. **缺 bundle 时整体失败、静态不伪造通过**：静态阶段如实 passed，
   质量门禁阶段 failed 且 ``bundle`` 输入为 ``false``。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

import yaml

from ai_detector.config import PROJECT_ROOT
from project_gate.isolated_check import (
    StageError,
    build_isolated_copy,
    create_isolated_venv,
    inside_isolated_copy,
    probe_resolution,
    read_report,
    run_in_copy,
    safe_stage_path,
)

#: 在**隔离副本内部**运行本模块时整体跳过：那时"真实仓库 → 副本"这层关系已经成立，
#: 再嵌套一层既无意义也不可行（``python -m venv`` 不能从 venv 内再建 venv）。
#: 真实仓库里的默认测试运行不受影响（判据是隔离 venv 独有的标记文件）。
pytestmark = pytest.mark.skipif(
    inside_isolated_copy(),
    reason="已在 CW-013 的隔离副本内：不再嵌套生成副本与 venv",
)

STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests" / "cw013"

#: 本模块自己的暂存区（项目相对）：必须是 ``artifacts/tmp_tests/`` 的真子目录。
STAGE_AREA_REL = "artifacts/tmp_tests"
STAGE_ROOT_REL = "artifacts/tmp_tests/cw013"

CW009_REPORT_REL = "artifacts/reports/cw009_quality_gate/report.json"
GATE_CONFIG_REL = "configs/quality_gate.yaml"

#: 真实工作目录中的冻结文件：本轮验证不得改动它们。
FROZEN_REL = (
    "artifacts/models/cw004/lightgbm_bundle.joblib",
    "configs/quality_gate.yaml",
    "tests/fixtures/cw009/model_regression_cases.json",
)
DEFAULT_DB_REL = "artifacts/runtime/cw007_decisions.sqlite3"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_snapshot() -> dict[str, str | None]:
    snapshot = {relative: _digest(PROJECT_ROOT / relative) for relative in FROZEN_REL}
    default_db = PROJECT_ROOT / DEFAULT_DB_REL
    snapshot[DEFAULT_DB_REL] = _digest(default_db) if default_db.is_file() else None
    return snapshot


@pytest.fixture(scope="module")
def isolated() -> dict:
    """建一次隔离副本 + 独立 venv，模块内共用；结束后清理并复核真实产物未变。"""
    before = _frozen_snapshot()
    directory = STAGE_ROOT / uuid.uuid4().hex
    copy_root = directory / "checkout"
    build_isolated_copy(PROJECT_ROOT, copy_root)
    interpreter = create_isolated_venv(copy_root)
    try:
        yield {"copy": copy_root, "python": interpreter, "stage": directory}
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    after = _frozen_snapshot()
    assert before == after, "隔离验证不得改动真实工作目录的冻结文件"


# --- 暂存路径边界：递归删除前必须限定（首次验收阻断项 3） --------------


@pytest.mark.parametrize(
    "stage",
    [
        "configs/cw013_stage",
        "src/cw013_stage",
        "artifacts",
        "artifacts/delivery/cw013_stage",
        "artifacts/models",
        ".",
        "..",
        "artifacts/tmp_tests/../../escape",
        "artifacts\\tmp_tests\\backslash",
    ],
)
def test_stage_path_is_restricted_to_the_ignored_tmp_area(stage: str) -> None:
    """只有项目内被忽略的 ``artifacts/tmp_tests/`` 之下的路径才被接受。"""
    with pytest.raises(StageError) as caught:
        safe_stage_path(stage, root=PROJECT_ROOT)
    assert caught.value.code == "stage_not_ignored"


def test_absolute_stage_path_is_rejected() -> None:
    with pytest.raises(StageError) as caught:
        safe_stage_path(str(PROJECT_ROOT / STAGE_ROOT_REL / "abs"), root=PROJECT_ROOT)
    assert caught.value.code == "stage_not_ignored"


def test_normal_stage_path_is_accepted() -> None:
    resolved = safe_stage_path(STAGE_ROOT_REL, root=PROJECT_ROOT)
    assert resolved == (PROJECT_ROOT / STAGE_ROOT_REL).resolve()


def test_stage_area_root_itself_is_rejected() -> None:
    """``artifacts/tmp_tests`` 本身不是"本轮专用目录"，不得作为删除目标。"""
    with pytest.raises(StageError) as caught:
        safe_stage_path(STAGE_AREA_REL, root=PROJECT_ROOT)
    assert caught.value.code == "stage_not_ignored"


def test_build_isolated_copy_refuses_an_unignored_stage() -> None:
    """指向未忽略目录时**不得执行任何递归删除**：留下哨兵文件作为证据。"""
    victim = PROJECT_ROOT / "artifacts" / "delivery" / f"cw013_victim_{uuid.uuid4().hex}"
    victim.mkdir(parents=True)
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    try:
        with pytest.raises(StageError):
            build_isolated_copy(PROJECT_ROOT, victim / "checkout")
        assert marker.is_file() and marker.read_text(encoding="utf-8") == "keep", (
            "不得删除未授权的目录"
        )
    finally:
        shutil.rmtree(victim, ignore_errors=True)


# --- 副本内容 ------------------------------------------------------------


def test_isolated_copy_carries_no_runtime_artifacts(isolated: dict) -> None:
    """副本必须**不带** bundle、默认库与任何报告——这才叫"干净检出"。"""
    copy_root: Path = isolated["copy"]
    assert not (copy_root / "artifacts").exists()
    for absent in (
        "artifacts/models/cw004/lightgbm_bundle.joblib",
        "artifacts/runtime/cw007_decisions.sqlite3",
        "artifacts/reports",
    ):
        assert not (copy_root / absent).exists(), absent
    for present in ("pyproject.toml", "conftest.py", "src", "scripts", "project_gate",
                    "configs", "tests", "performance"):
        assert (copy_root / present).exists(), present


def test_isolated_copy_covers_the_whole_static_scope(isolated: dict) -> None:
    """副本必须覆盖 ``STATIC_PATHS`` 的每一项，否则静态阶段会把"副本不完整"报成 E902 违规。"""
    from project_gate.static_checks import STATIC_PATHS

    copy_root: Path = isolated["copy"]
    for relative in STATIC_PATHS:
        assert (copy_root / relative).exists(), relative


# --- 解析路径 ------------------------------------------------------------


def test_module_resolution_points_into_the_copy(isolated: dict) -> None:
    """核心断言：项目代码从副本解析，且**不在**真实仓库的 ``src`` 之下。"""
    resolution = probe_resolution(isolated["python"], isolated["copy"])
    assert resolution.get("ok"), resolution
    assert resolution["module_inside_copy"], resolution
    assert resolution["root_is_copy"], resolution
    assert resolution["outside_real_repo"], resolution
    assert str(PROJECT_ROOT / "src") not in resolution["module"]
    assert Path(resolution["project_root"]) == isolated["copy"].resolve()


def test_isolated_interpreter_is_not_the_host_interpreter(isolated: dict) -> None:
    """隔离解释器必须是副本自己的 venv，而不是宿主解释器。"""
    assert Path(isolated["python"]).resolve() != Path(sys.executable).resolve()
    assert isolated["copy"].resolve() in Path(isolated["python"]).resolve().parents


def test_host_editable_finder_does_not_leak_into_the_copy(isolated: dict) -> None:
    """宿主的 editable 安装不得在隔离解释器里注册：那里的 ``ai_detector`` 只来自副本。"""
    probe = (
        "import json,sys;"
        "print(json.dumps([p for p in sys.path if 'site-packages' in p]))"
    )
    completed = run_in_copy(isolated["copy"], isolated["python"], ["-c", probe])
    assert completed.returncode == 0, completed.stdout
    paths = json.loads(completed.stdout.strip().splitlines()[-1])
    # 宿主 site-packages 只作为**普通路径**存在，其中的 .pth 不会被处理
    assert all("__editable__" not in item for item in paths)


# --- 缺 bundle：预检失败关闭，静态不伪造通过 -----------------------------


def test_missing_bundle_makes_the_isolated_gate_fail(isolated: dict) -> None:
    """缺 bundle：总门禁退出 5，静态通过、质量门禁失败、bundle 输入为 false。"""
    copy_root: Path = isolated["copy"]
    completed = run_in_copy(
        copy_root, isolated["python"], ["scripts/run_project_gate.py"], timeout_s=1800
    )
    report = read_report(copy_root, "artifacts/reports/cw012_project_gate/report.json")
    assert report is not None, completed.stdout[-800:]
    assert completed.returncode == 5, completed.stdout[-800:]
    assert report["status"] == "failed" and report["exit_code"] == 5

    static = report["phases"]["static"]
    assert static["status"] == "passed", "静态阶段应如实通过"
    assert static["violations"] == 0

    quality = report["phases"]["quality_gate"]
    assert quality["status"] == "failed"
    assert quality["exit_code"] == 2
    assert quality["cw009_status"] == "failed"
    assert quality["inputs_match"]["bundle"] is False
    for label in ("service_config", "policy", "corpus"):
        assert quality["inputs_match"][label] is True, label


def test_isolated_preflight_failure_reports_the_bundle_as_missing(isolated: dict) -> None:
    """再看 CW-009 自己的报告：``input_missing``，且回归与 pytest 都不运行。"""
    copy_root: Path = isolated["copy"]
    completed = run_in_copy(
        copy_root,
        isolated["python"],
        ["scripts/run_quality_gate.py", "--config", GATE_CONFIG_REL, "--regression-only"],
        timeout_s=900,
    )
    report = read_report(copy_root, CW009_REPORT_REL)
    assert report is not None, completed.stdout[-800:]
    assert completed.returncode == 2
    assert report["overall_status"] == "failed"
    assert report["error_code"] == "input_missing"
    assert report["inputs"]["bundle"]["match"] is False
    assert report["inputs"]["bundle"]["actual_sha256"] is None
    assert report["pytest"] == {"status": "not_run"}
    assert not report["regression"]


def test_isolated_report_leaks_no_absolute_paths(isolated: dict) -> None:
    """隔离运行写出的报告不得含绝对路径（副本路径或真实路径都不行）。"""
    copy_root: Path = isolated["copy"]
    raw = (copy_root / "artifacts/reports/cw012_project_gate/report.json").read_text(
        encoding="utf-8"
    )
    assert str(copy_root) not in raw
    assert str(PROJECT_ROOT) not in raw
    assert "::" not in raw, "不得含 pytest 节点 ID"
    json.loads(raw)  # 必须是合法 JSON


# --- 在隔离解释器里证明导出/导入可用（用合成产物，不依赖真实 bundle） ----


def test_export_and_import_work_inside_the_isolated_interpreter(isolated: dict) -> None:
    """在副本的隔离解释器里跑一次完整的"导出 → 删目标 → 导入 → 复核字节"。

    用**合成**的小 bundle 与配套配置，因此在任何深度的隔离副本里都成立，
    不依赖真实 bundle 是否存在。
    """
    copy_root: Path = isolated["copy"]
    stage = copy_root / "artifacts" / "tmp_tests" / "cw013_probe"
    stage.mkdir(parents=True, exist_ok=True)
    payload = b"cw013-isolated-synthetic-" + b"z" * 48
    bundle = stage / "model" / "bundle.joblib"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    config = stage / "quality_gate.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "quality_gate": {
                    "inputs": {
                        "bundle": {"path": f"{stage.relative_to(copy_root).as_posix()}/model/bundle.joblib",
                                   "sha256": digest}
                    },
                    "report": {"path": f"{stage.relative_to(copy_root).as_posix()}/report.json"},
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    program = (
        "import json,sys;"
        "from pathlib import Path;"
        "from project_gate import bundle_delivery as bd;"
        f"root=Path(r'{copy_root}');"
        f"cfg=r'{config.relative_to(copy_root).as_posix()}';"
        f"pkg=r'artifacts/delivery/cw013_probe_package';"
        "exported=bd.export_bundle(root=root, package_dir=pkg, config_rel=cfg);"
        "(root/Path(bd.frozen_target(root, cfg).relative_path)).unlink();"
        "imported=bd.import_bundle(root=root, package_dir=pkg, config_rel=cfg);"
        "print(json.dumps({'export': exported['status'], 'import': imported['status'],"
        "'target_bytes': (root/Path(bd.frozen_target(root, cfg).relative_path)).read_bytes().decode()}))"
    )
    completed = run_in_copy(copy_root, isolated["python"], ["-c", program], timeout_s=300)
    assert completed.returncode == 0, completed.stdout[-800:]
    outcome = json.loads(completed.stdout.strip().splitlines()[-1])
    assert outcome["export"] == "exported"
    assert outcome["import"] == "imported"
    assert outcome["target_bytes"] == payload.decode()
    assert (copy_root / "artifacts/delivery/cw013_probe_package" / "manifest.json").is_file()
    assert not (stage / "package").exists(), "合成包不得落在 tmp_tests 下"
