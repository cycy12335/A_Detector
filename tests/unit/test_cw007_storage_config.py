"""CW-007 存储配置的单元测试：严格 storage 段、冻结版本、路径被限制在安全目录。

三条最要紧的性质：

1. **storage 段严格**：只允许 ``path`` 与 ``schema_version`` 两个字段，版本必须是冻结值；
2. **路径被限制在项目相对 ``artifacts/runtime/*.sqlite3``**：绝对路径、盘符、UNC、扩展长度
   前缀、根相对、``..`` 逃逸、疑似凭据、错误目录、错误扩展名一律拒绝，且判定与平台无关；
3. **符号链接/重解析点不得逃出项目根**：已在磁盘上存在的组件若通过链接指向外部，加载阶段
   失败关闭。
"""

from __future__ import annotations

import os
import shutil
import stat as stat_module
from pathlib import Path

import pytest

from ai_detector.config import PROJECT_ROOT
from ai_detector.service.config import ServiceConfig, ServiceConfigError
from ai_detector.storage.config import (
    DEFAULT_STORAGE_PATH,
    LEGACY_STORAGE_SCHEMA_VERSION,
    LEGACY_USER_VERSION,
    RECORD_SCHEMA_VERSION,
    STORAGE_SCHEMA_VERSION,
    STORAGE_SUFFIX,
    USER_VERSION,
    StorageErrorCode,
    StorageLoadError,
    assert_safe_storage_path,
    resolve_storage_path,
)
from tests.cw007_fixtures import storage_section, staged_db_relative

SERVICE_CONFIG_PATH = PROJECT_ROOT / "configs" / "service.yaml"


# --- 冻结常量与仓库配置 -------------------------------------------------


def test_frozen_versions_are_the_documented_values() -> None:
    """CW-008 把 storage schema 提升为 v2 并冻结 legacy v1；记录 schema 与路径**未变**。"""
    assert STORAGE_SCHEMA_VERSION == "cw008_storage_v2"
    assert LEGACY_STORAGE_SCHEMA_VERSION == "cw007_storage_v1"
    assert RECORD_SCHEMA_VERSION == "cw007_decision_record_v1"
    assert (USER_VERSION, LEGACY_USER_VERSION) == (2, 1)
    assert STORAGE_SUFFIX == ".sqlite3"
    assert DEFAULT_STORAGE_PATH == "artifacts/runtime/cw007_decisions.sqlite3"


def test_repository_config_declares_the_frozen_storage_section() -> None:
    config = ServiceConfig.from_yaml(SERVICE_CONFIG_PATH)
    assert config.storage_relative_path == DEFAULT_STORAGE_PATH
    assert config.storage_schema_version == STORAGE_SCHEMA_VERSION
    assert config.storage_path is not None
    assert config.storage_path.is_absolute()
    assert PROJECT_ROOT in config.storage_path.parents


def test_config_parsing_never_creates_a_database(tmp_path: Path) -> None:
    """配置解析**不得**产生任何文件 I/O。

    用一个独一无二的临时路径来断言，而不是断言"生产默认库不存在"——后者会被
    真实进程测试（它们用仓库配置启动 CLI）合法地创建出来，属于全局状态依赖。
    """
    relative = staged_db_relative(tmp_path)
    path = PROJECT_ROOT / relative
    assert not path.exists()
    config = _from({"path": relative, "schema_version": STORAGE_SCHEMA_VERSION})
    assert config.storage_relative_path == relative
    assert not path.exists(), "配置解析不得创建数据库文件"


def test_config_parsing_does_not_touch_an_existing_database() -> None:
    """即使生产库已存在，解析配置也不得改写它（不建表、不迁移、不打开）。"""
    path = PROJECT_ROOT / DEFAULT_STORAGE_PATH
    before = path.stat().st_mtime_ns if path.exists() else None
    ServiceConfig.from_yaml(SERVICE_CONFIG_PATH)
    if before is None:
        assert not path.exists(), "配置解析不得创建数据库"
    else:
        assert path.stat().st_mtime_ns == before, "配置解析不得触碰已有数据库"


# --- 严格 storage 段 ----------------------------------------------------


def _mapping(storage: object) -> dict:
    raw = {
        "service": {
            "name": "svc",
            "api_version": "v1",
            "model_version": "m1",
            "host": "127.0.0.1",
            "port": 8000,
            "batch_limit": 100,
            "request_id_header": "X-Request-ID",
            "score_semantics": "uncalibrated_model_score",
            "threshold_policy": "validation_f1",
            "expected_source_features": 199,
        },
        "bundle": {
            "path": "artifacts/models/cw004/lightgbm_bundle.joblib",
            "sha256": "a" * 64,
            "expected_schema_version": "cw004_bundle_v1",
        },
        "limits": {"max_request_id_length": 64, "request_id_pattern": "^[A-Za-z0-9._:-]{1,64}$"},
    }
    if storage is not None:
        raw["storage"] = storage
    return raw


def _from(storage: object) -> ServiceConfig:
    return ServiceConfig.from_mapping(_mapping(storage), config_path=Path("probe.yaml"))


def test_storage_section_is_optional() -> None:
    """未配置 storage 是合法状态：应用仍启动，只让 decisions 接口降级。"""
    config = _from(None)
    assert config.storage_path is None
    assert config.storage_relative_path is None
    assert config.storage_public_dict() == {
        "storage_configured": False,
        "storage_relative_path": None,
        "storage_schema_version": None,
    }


@pytest.mark.parametrize("key", ["path", "schema_version"])
def test_missing_storage_field_is_rejected(key: str) -> None:
    section = storage_section(DEFAULT_STORAGE_PATH)
    section.pop(key)
    with pytest.raises(ServiceConfigError, match="storage 段缺少字段"):
        _from(section)


@pytest.mark.parametrize(
    "extra", [{"extra": 1}, {"busy_timeout": 2000}, {"path": DEFAULT_STORAGE_PATH, "journal": "WAL"}]
)
def test_extra_storage_field_is_rejected(extra: dict) -> None:
    """多一个字段就拒绝——不允许"看起来被接受、实际被忽略"的配置。"""
    section = {**storage_section(DEFAULT_STORAGE_PATH), **extra}
    with pytest.raises(ServiceConfigError, match="storage 段"):
        _from(section)


def test_storage_must_be_a_mapping() -> None:
    with pytest.raises(ServiceConfigError, match="storage 段"):
        _from([DEFAULT_STORAGE_PATH])


@pytest.mark.parametrize(
    "version",
    ["cw007_storage_v2", "cw006_storage_v1", "", "CW007_STORAGE_V1", None, 1],
)
def test_non_canonical_storage_schema_version_is_rejected(version: object) -> None:
    with pytest.raises(ServiceConfigError, match="storage.schema_version"):
        _from({"path": DEFAULT_STORAGE_PATH, "schema_version": version})


# --- 路径形状：平台无关的拒绝 -------------------------------------------


@pytest.mark.parametrize(
    "unsafe",
    [
        "C:\\elsewhere\\cw007.sqlite3",
        "D:/elsewhere/cw007.sqlite3",
        "\\\\server\\share\\cw007.sqlite3",
        "\\\\?\\C:\\elsewhere\\cw007.sqlite3",
        "/etc/cw007.sqlite3",
        "\\cw007.sqlite3",
        "../outside/cw007.sqlite3",
        "artifacts/runtime/../../outside.sqlite3",
        "artifacts/policy_token.sqlite3",
        "artifacts/cw007.sqlite3",
        "artifacts/runtime/cw007_decisions.db",
        "artifacts/runtime/nested/cw007.sqlite3",
    ],
)
def test_config_rejects_unsafe_storage_paths(unsafe: str) -> None:
    """配置层（``ServiceConfig``）必须把自己的错误类型暴露出来，CLI 才能以退出码 2 失败。"""
    with pytest.raises(ServiceConfigError) as error:
        _from({"path": unsafe, "schema_version": STORAGE_SCHEMA_VERSION})
    assert "storage" in str(error.value)


@pytest.mark.parametrize(
    "unsafe",
    [
        "C:\\elsewhere\\cw007.sqlite3",
        "\\\\server\\share\\cw007.sqlite3",
        "\\\\?\\C:\\elsewhere\\cw007.sqlite3",
        "/etc/cw007.sqlite3",
        "\\cw007.sqlite3",
        "../outside/cw007.sqlite3",
        "artifacts/runtime/../../outside.sqlite3",
        "artifacts/policy_token.sqlite3",
    ],
)
def test_low_level_path_guard_rejects_the_same_shapes(unsafe: str) -> None:
    with pytest.raises(StorageLoadError) as error:
        assert_safe_storage_path(unsafe)
    assert error.value.code == StorageErrorCode.PATH_UNSAFE


@pytest.mark.parametrize(
    "outside_dir",
    [
        "artifacts/cw007.sqlite3",  # 不在 runtime 下
        "artifacts/models/cw007.sqlite3",
        "data/raw/cw007.sqlite3",
        "cw007.sqlite3",
        "configs/cw007_decisions.sqlite3",
    ],
)
def test_paths_outside_the_runtime_directory_are_rejected(outside_dir: str) -> None:
    with pytest.raises(StorageLoadError) as error:
        assert_safe_storage_path(outside_dir)
    assert error.value.code == StorageErrorCode.PATH_UNSAFE


@pytest.mark.parametrize(
    "wrong_suffix",
    [
        "artifacts/runtime/cw007_decisions.db",
        "artifacts/runtime/cw007_decisions.sqlite",
        "artifacts/runtime/cw007_decisions",
        "artifacts/runtime/cw007_decisions.sqlite3.txt",
        "artifacts/runtime/cw007_decisions.SQLITE3",
    ],
)
def test_wrong_extension_is_rejected(wrong_suffix: str) -> None:
    with pytest.raises(StorageLoadError) as error:
        assert_safe_storage_path(wrong_suffix)
    assert error.value.code == StorageErrorCode.PATH_UNSAFE


@pytest.mark.parametrize(
    "nested",
    [
        "artifacts/runtime/nested/cw007.sqlite3",
        "artifacts/runtime/./cw007.sqlite3",
    ],
)
def test_paths_with_extra_components_are_rejected(nested: str) -> None:
    """只允许 ``artifacts/runtime/<file>.sqlite3`` 这一种形状，不接受子目录或 ``.`` 组件。"""
    with pytest.raises(StorageLoadError) as error:
        assert_safe_storage_path(nested)
    assert error.value.code == StorageErrorCode.PATH_UNSAFE


def test_repository_default_path_is_accepted() -> None:
    assert assert_safe_storage_path(DEFAULT_STORAGE_PATH) == DEFAULT_STORAGE_PATH
    assert resolve_storage_path(DEFAULT_STORAGE_PATH) == PROJECT_ROOT / DEFAULT_STORAGE_PATH


def test_test_database_path_shape_is_accepted(tmp_path: Path) -> None:
    """应用级测试用的路径同样落在生产允许的形状内（不为测试放宽入口）。"""
    relative = staged_db_relative(tmp_path)
    assert assert_safe_storage_path(relative) == relative


# --- 符号链接/重解析点逃逸：失败关闭 ------------------------------------
#
# 判定逻辑用**注入式 realpath** 做确定性覆盖（不依赖平台是否允许创建符号链接），
# 再在允许创建链接的平台上真机复验一次——两者都**不跳过**，因此 CW-007 专项零跳过。


def test_resolved_path_escaping_the_project_is_rejected(monkeypatch, tmp_path: Path) -> None:
    """解析后的真实路径落在项目根之外时，必须失败关闭。"""
    from ai_detector.storage import config as storage_config

    outside = tmp_path / "outside_runtime"
    outside.mkdir(parents=True, exist_ok=True)
    real_resolver = storage_config.resolve_real_path

    def _escaping(path):  # noqa: ANN001, ANN202
        # 只劫持数据库路径的解析；项目根本身保持真实行为。
        if str(path).endswith(".sqlite3"):
            return outside / Path(path).name
        return real_resolver(path)

    monkeypatch.setattr(storage_config, "resolve_real_path", _escaping)
    with pytest.raises(StorageLoadError) as error:
        storage_config.resolve_storage_path(DEFAULT_STORAGE_PATH)
    assert error.value.code == StorageErrorCode.PATH_UNSAFE


def test_resolved_path_staying_inside_the_project_is_accepted(monkeypatch) -> None:
    """对照组：解析仍在项目根之内时照常通过（证明上面的拒绝不是"一律拒绝"）。"""
    from ai_detector.storage import config as storage_config

    calls = {"n": 0}
    real_resolver = storage_config.resolve_real_path

    def _counting(path):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return real_resolver(path)

    monkeypatch.setattr(storage_config, "resolve_real_path", _counting)
    resolved = storage_config.resolve_storage_path(DEFAULT_STORAGE_PATH)
    assert resolved == PROJECT_ROOT / DEFAULT_STORAGE_PATH
    assert calls["n"] == 2, "项目根与候选路径都应被解析比较"


def _try_directory_link(link: Path, target: Path) -> bool:
    """尽力创建一个目录链接（Windows 用联接点，无需特权；POSIX 用符号链接）。

    Returns:
        ``True`` 表示已创建，``False`` 表示当前平台/权限不支持。
    """
    import subprocess

    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and link.exists()
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    return True


def _link_entry_present(link: Path) -> bool:
    """链接条目是否仍在（**包括**目标已消失的悬挂链接）。

    ``Path.exists()`` 对悬挂链接返回 ``False``，因此不能用它判断残留。
    """
    try:
        link.lstat()
    except OSError:
        return False
    return True


def _remove_directory_link(link: Path) -> None:
    """删除目录链接**条目本身**，绝不递归进目标。

    Windows 的 ``mklink /J`` 产生的是**联接点**：``Path.is_symlink()`` 对它返回 ``False``，
    而目标一旦消失，连 ``Path.exists()`` 也返回 ``False``——只按这两者判断会漏删，把悬挂
    联接点留在 ``artifacts/`` 里。因此这里按 reparse 属性识别，并用 ``os.rmdir`` 删除
    （只摘掉链接，不触碰目标内容；``shutil.rmtree`` 会先递归进目标，对联接点不安全）。
    """
    try:
        attributes = getattr(link.lstat(), "st_file_attributes", 0)
    except OSError:
        return
    is_reparse = bool(
        attributes & getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if os.path.islink(link):
        os.unlink(link)
    elif is_reparse:
        os.rmdir(link)
    elif link.is_dir():
        shutil.rmtree(link, ignore_errors=True)


def test_real_directory_link_escaping_the_project_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    """真机复验：``artifacts/<link>/x.sqlite3`` 经链接解析逃出项目根时必须失败关闭。

    链接无法创建时（受限的 Windows 权限）改用注入式解析覆盖**同一条**判定，因此本用例
    在任何平台上都真实执行、**不跳过**。
    """
    from ai_detector.storage import config as storage_config

    link = PROJECT_ROOT / "artifacts" / f"cw007_escape_{tmp_path.name}"
    outside = tmp_path / "outside_runtime"
    outside.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = f"artifacts/{link.name}/cw007.sqlite3"
    created = _try_directory_link(link, outside)
    try:
        if not created:
            real_resolver = storage_config.resolve_real_path
            monkeypatch.setattr(
                storage_config,
                "resolve_real_path",
                lambda path: outside
                if str(path).endswith(".sqlite3")
                else real_resolver(path),
            )
        with pytest.raises(StorageLoadError) as error:
            storage_config.resolve_storage_path(relative)
        assert error.value.code == StorageErrorCode.PATH_UNSAFE
    finally:
        _remove_directory_link(link)
    # 测试自己创建的链接**必须**清理干净：本项目的工作树里不允许留下任何残留条目，
    # 悬挂的联接点也不行（`Path.exists()` 对它是 False，所以要按 lstat 判定）。
    assert not _link_entry_present(link), "测试未清理自己创建的目录链接"
