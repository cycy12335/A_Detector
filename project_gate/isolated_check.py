"""CW-013 隔离源码副本：证明门禁能在不携带 ``artifacts/**`` 的副本里真实运行。

## 为什么不能只靠 ``sys.path``

CW-012 发现本项目以 **editable 方式**安装在 ``.venv``：``__editable__.ai_detector-0.1.0.pth``
注册的是 **MetaPathFinder**，它的优先级高于 ``sys.path``，因此无论怎么设 ``PYTHONPATH``，
``import ai_detector`` 都会解析到真实仓库、``PROJECT_ROOT`` 也永远是真实项目根。

## 本模块的办法：一个**不注册任何 finder** 的独立解释器

新建一个 venv，只放两个**普通路径** ``.pth``：

1. ``_cw013_copy_src.pth`` → ``<副本>/src``：项目代码从副本解析
   （``config.py`` 的 ``parents[2]`` 因此等于副本根，``PROJECT_ROOT`` 正确）；
2. ``_cw013_host_deps.pth`` → 宿主 venv 的 ``site-packages``：第三方依赖可导入。

关键在于**宿主 venv 的 ``.pth`` 文件不会被处理**——``site`` 只为"当前解释器的 site 目录"
处理 ``.pth``，而被当成普通路径加进来的目录不在此列。因此宿主的 editable finder
**不会注册**，不可能回指真实仓库。副本里也**没有**任何形式的包安装。

实测：新 venv 里 ``import ai_detector`` 在放 ``.pth`` 之前是 ``ModuleNotFoundError``，
放上 ``_cw013_copy_src.pth`` 之后解析到副本；``ai_detector.config.PROJECT_ROOT`` 等于副本根。

## 诚实的限制

- 隔离的是**代码解析**，不是**依赖**：第三方包复用宿主 venv 里已安装的同一版本，
  没有重新下载或冻结。要连依赖一起隔离，需要另一个包管理决定（见
  ``docs/bundle_delivery.md``）。
- 副本是"已提交源码的副本"，**不是**真实 Git checkout（当前没有 ``.git``），
  也不代表 hosted CI。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

#: 复制进隔离副本的**已提交**条目（相对项目根）。
#: 必须覆盖 ``static_checks.STATIC_PATHS`` 里的每一项，否则 ruff 会报 E902（路径不存在）
#: 而把"副本不完整"误判成"静态有违规"。
ISOLATED_COPY_ENTRIES: tuple[str, ...] = (
    ".gitignore",
    "pyproject.toml",
    "conftest.py",
    "src",
    "scripts",
    "project_gate",
    "configs",
    "tests",
    "performance",
    "docs",
    "constraints",
    ".github",
)

#: 除 bundle 之外，**全量测试套件**还需要一个被 ``.gitignore`` 排除的本地产物：
#: ``artifacts/models/cw003/rule_baseline_state.json``（``test_cw006_policy_config.py``
#: 用它核对政策阈值与 CW-003 透明规则状态一致）。
#:
#: CW-013 的隔离验证发现全量套件当时还需要**默认库** ``cw007_decisions.sqlite3``；
#: CW-014 按验收标准把"默认库不被改写"改成了会话级守卫
#: （根 ``conftest.py::guard_default_database``），因此默认库**不再**是必需输入，
#: 本元组也随之只剩一项。CW-003 状态由 :data:`CW014_FIXTURE` 指向的已审查夹具供给。
EXTRA_IGNORED_INPUTS: tuple[str, ...] = ("artifacts/models/cw003/rule_baseline_state.json",)

#: 供给 CW-003 状态的项目相对**已审查夹具**（随源码提交，见 `configs/ci_inputs.yaml`）。
CW014_FIXTURE: str = "tests/fixtures/cw014/rule_baseline_state.json"

#: 绝不复制进副本的条目：运行产物、虚拟环境与版本控制元数据。
NEVER_COPIED: tuple[str, ...] = ("artifacts", ".venv", ".git", ".pytest_cache")

#: 隔离 venv 里两个路径 .pth 的文件名。
COPY_SRC_PTH = "_cw013_copy_src.pth"
HOST_DEPS_PTH = "_cw013_host_deps.pth"

#: 隔离副本里必须**不存在**、用于证明"没带运行产物"的路径。
MUST_BE_ABSENT = (
    "artifacts/models/cw004/lightgbm_bundle.joblib",
    "artifacts/runtime/cw007_decisions.sqlite3",
    "artifacts/reports",
)

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")

#: 暂存区必须位于项目内被 Git 忽略的这个子树之下。
STAGE_ROOT_REL = "artifacts/tmp_tests"


class StageError(RuntimeError):
    """暂存路径越界（稳定分类 ``stage_not_ignored``）。"""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def assert_inside_stage_area(path: Path, *, root: Path) -> Path:
    """校验一个**已解析**路径落在项目内被忽略的 ``artifacts/tmp_tests/`` 之下。

    这是 ``rmtree`` 之前的最后一道闸：无论调用方怎么算出这个路径，只要它不在被忽略的
    暂存区里就拒绝。绝对路径在这里是允许的（调用方已完成输入校验），
    被拒的是**位置**而不是写法——``<root>/configs/...`` 之类一律拒绝。

    Raises:
        StageError: 路径等于暂存区根，或不在其下。
    """
    resolved = Path(path).resolve()
    area = (root.resolve() / STAGE_ROOT_REL).resolve()
    if resolved == area or area not in resolved.parents:
        raise StageError(
            "stage_not_ignored",
            f"暂存路径必须位于 {STAGE_ROOT_REL}/ 之下，且不是它本身",
        )
    return resolved


def safe_stage_path(stage: str | Path, *, root: Path) -> Path:
    """把命令行给出的暂存路径限定在项目内被忽略的 ``artifacts/tmp_tests/`` 之下。

    这一步必须在**任何递归删除之前**完成：``--stage`` 未校验时会被直接拼到项目根，
    绝对路径甚至会覆盖项目根，随后 ``shutil.rmtree(<stage>/checkout)`` 就可能对
    项目外的位置执行破坏性删除。

    只接受**项目相对**路径，且必须是 ``artifacts/tmp_tests`` 的**真子目录**
    （既不是它本身，也不是它之外的任何位置）。绝对路径、盘符、反斜杠、``..``、
    ``.``、连续斜杠与符号链接一律拒绝。

    Raises:
        StageError: 写法不合规或解析后不在 ``artifacts/tmp_tests/`` 之下。
    """
    text = str(stage)
    if not text:
        raise StageError("stage_not_ignored", "暂存路径为空")
    candidate = Path(text)
    if candidate.is_absolute() or text.startswith(("/", "\\")) or ":" in text or "\\" in text:
        raise StageError("stage_not_ignored", "暂存路径必须是项目相对路径，且不含盘符或反斜杠")
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise StageError("stage_not_ignored", "暂存路径不得含空、. 或 .. 组件")

    resolved_root = root.resolve()
    cursor = resolved_root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise StageError("stage_not_ignored", "暂存路径组件不得是符号链接")
    return assert_inside_stage_area(cursor, root=root)


def build_isolated_copy(root: Path, stage: Path) -> Path:
    """把已提交源码复制到 ``stage``，并断言没有携带任何运行产物。

    ``stage`` 先经 :func:`assert_inside_stage_area` 限定；因此这里的 ``rmtree`` 只可能
    发生在项目内被忽略的暂存区里，不会碰项目外、已提交目录或既有的非本轮目录。
    """
    stage = assert_inside_stage_area(stage, root=root)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for entry in ISOLATED_COPY_ENTRIES:
        source = root / entry
        if not source.exists():
            continue
        target = stage / entry
        if source.is_dir():
            shutil.copytree(source, target, ignore=_IGNORE)
        else:
            shutil.copy2(source, target)
    for forbidden in NEVER_COPIED:
        assert not (stage / forbidden).exists(), f"{forbidden} 不应出现在隔离副本里"
    for relative in MUST_BE_ABSENT:
        assert not (stage / relative).exists(), f"{relative} 不应出现在隔离副本里"
    return stage


def create_isolated_venv(copy_root: Path, *, host_site: Path | None = None) -> Path:
    """在副本内建独立 venv，并让它把副本当作项目代码来源。

    Returns:
        隔离解释器的路径（``<copy>/.venv/Scripts/python.exe`` 或 ``bin/python``）。
    """
    venv_root = copy_root / ".venv"
    completed = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_root)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if completed.returncode != 0:  # pragma: no cover - 环境级失败
        raise RuntimeError(f"venv 创建失败：{completed.stderr.strip()[-200:]}")

    site = _venv_site_packages(venv_root)
    host = host_site or Path(sys.prefix) / "Lib" / "site-packages"
    if not host.is_dir():  # pragma: no cover - POSIX 布局
        host = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    # 普通路径 .pth：只把目录加进 sys.path，不注册任何 finder，
    # 因此宿主 venv 的 .pth（含 editable finder）不会被处理。
    (site / HOST_DEPS_PTH).write_text(f"{host}\n", encoding="utf-8")
    (site / COPY_SRC_PTH).write_text(f"{copy_root / 'src'}\n", encoding="utf-8")
    _link_host_tools(venv_root)
    return venv_python(venv_root)


def _link_host_tools(venv_root: Path) -> None:
    """把宿主 venv 里的**命令行工具**（ruff）复用到隔离 venv。

    ``python -m ruff`` 只是包装器，它按 ``sys.executable`` 旁边的 ``Scripts/`` 去找
    ``ruff.exe``。隔离 venv 里没有那个二进制，包装器会 ``FileNotFoundError``——
    于是静态阶段会失败（虽然失败关闭，但会被报成"有违规"，具有误导性）。
    这里把同一个锁定版本的二进制复制过去，隔离运行才有真实的静态结果。

    复用的是**同一个版本**（``pyproject.toml`` 用 ``==`` 锁定），没有重新下载。
    """
    for name in ("ruff.exe", "ruff"):
        source = Path(sys.prefix) / "Scripts" / name
        if not source.is_file():  # pragma: no cover - POSIX 布局
            source = Path(sys.prefix) / "bin" / name
        if source.is_file():
            shutil.copy2(source, venv_root / "Scripts" / name if (venv_root / "Scripts").is_dir() else venv_root / "bin" / name)
            return


def _venv_site_packages(venv_root: Path) -> Path:
    for candidate in (
        venv_root / "Lib" / "site-packages",
        venv_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ):
        if candidate.is_dir():
            return candidate
    raise RuntimeError("找不到隔离 venv 的 site-packages")  # pragma: no cover


def venv_python(venv_root: Path) -> Path:
    for candidate in (venv_root / "Scripts" / "python.exe", venv_root / "bin" / "python"):
        if candidate.exists():
            return candidate
    raise RuntimeError("找不到隔离 venv 的解释器")  # pragma: no cover


def site_packages(venv_root: Path) -> Path:
    """隔离 venv 的 site-packages（POSIX/Windows 两种布局）。"""
    return _venv_site_packages(venv_root)


def inside_isolated_copy() -> bool:
    """当前解释器是否运行在 CW-013 造的隔离副本里。

    判据是**结构性的**：隔离 venv 的 site-packages 里存在本模块写入的
    :data:`COPY_SRC_PTH` 标记。真实仓库的 venv 里没有它，因此默认测试运行不受影响。

    用途：隔离副本内的测试运行**不再嵌套**做隔离验证。嵌套既无意义（被验证的关系已经
    成立），又不可行——``python -m venv`` 无法从 venv 内部再建 venv
    （实测报 ``failed to locate pyvenv.cfg``）。
    """
    for candidate in (
        Path(sys.prefix) / "Lib" / "site-packages",
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ):
        if (candidate / COPY_SRC_PTH).is_file():
            return True
    return False


#: 在隔离解释器里打印解析结果；用 ``json`` 输出便于断言。
_PROBE = (
    "import json, pathlib, sys;"
    "import ai_detector, ai_detector.config as config;"
    "print(json.dumps({"
    "'module': str(pathlib.Path(ai_detector.__file__).resolve()),"
    "'project_root': str(pathlib.Path(config.PROJECT_ROOT).resolve()),"
    "'executable': sys.executable,"
    "}))"
)


def probe_resolution(venv_python_path: Path, copy_root: Path) -> dict:
    """在隔离解释器里取 ``ai_detector`` 与 ``PROJECT_ROOT`` 的实际解析路径。

    Returns:
        ``{"module", "project_root", "executable", "module_inside_copy",
           "root_is_copy", "outside_real_repo"}``；无法导入时 ``ok=False``。
    """
    completed = subprocess.run(
        [str(venv_python_path), "-c", _PROBE],
        cwd=copy_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "stderr_tail": completed.stderr.strip()[-300:],
        }
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    copy_resolved = copy_root.resolve()
    module = Path(payload["module"])
    root_path = Path(payload["project_root"])
    real_src = (Path(__file__).resolve().parents[1] / "src").resolve()
    return {
        "ok": True,
        "module": str(module),
        "project_root": str(root_path),
        "executable": payload["executable"],
        "module_inside_copy": copy_resolved in module.parents,
        "root_is_copy": root_path == copy_resolved,
        "outside_real_repo": real_src not in module.parents,
    }


def provide_extra_ignored_inputs(root: Path, copy_root: Path) -> list[str]:
    """把全量测试套件额外需要的被忽略产物放进副本，返回实际提供的相对路径。

    优先使用**已审查夹具**（``tests/fixtures/cw014/``）：夹具随源码提交，
    副本里本来就有，因此这条路径与 CI 完全一致（CI 由 ``scripts/ci_fetch_inputs.py``
    做同样的事）。只有当夹具不存在时才回退到从工作目录复制，并如实记录。
    """
    provided: list[str] = []
    for relative in EXTRA_IGNORED_INPUTS:
        fixture = copy_root / CW014_FIXTURE
        if relative.endswith("rule_baseline_state.json") and fixture.is_file():
            target = copy_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fixture, target)
            provided.append(f"{relative} (committed fixture)")
            continue
        source = root / relative
        if not source.is_file():
            continue
        target = copy_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provided.append(f"{relative} (copied from workspace)")
    return provided


def run_in_copy(
    copy_root: Path, venv_python_path: Path, argv: Sequence[str], *, timeout_s: float = 1800.0
) -> subprocess.CompletedProcess:
    """在副本里用隔离解释器运行命令（不使用 shell，cwd 固定在副本根）。"""
    return subprocess.run(
        [str(venv_python_path), *argv],
        cwd=copy_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )


def read_report(copy_root: Path, relative: str) -> dict | None:
    """读取副本里某个项目相对报告；不存在或不可解析时返回 ``None``。"""
    path = copy_root / relative
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
