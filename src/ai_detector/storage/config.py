"""CW-007 存储配置：冻结版本常量与**安全的项目相对**数据库路径。

## 冻结语义写在代码里

storage schema、record schema、``PRAGMA user_version`` 与 busy timeout 全部是模块常量，
配置文件只能**复述**它们。数据库路径被限制成唯一一种形状：
``artifacts/runtime/<文件名>.sqlite3``——因此"把库写到别处"不是配置问题，而是加载失败。

## 为什么路径校验分成两段

- :func:`assert_safe_storage_path` 是**纯字符串**判定（平台无关的形状检查），因此可以在
  配置解析阶段调用——配置解析与 ``create_app()`` **不做文件 I/O**；
- :func:`resolve_storage_path` 额外做**符号链接/重解析点逃逸**检查（需要 ``realpath``，是 I/O），
  因此只在 app lifespan 的存储初始化阶段调用。

两段合起来仍然保证：在任何**创建**动作之前，路径要么是安全的，要么就已经失败关闭。
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ai_detector.config import PROJECT_ROOT, resolve_path

#: 当前 storage schema 版本（同时写入 ``storage_metadata`` 表）。
STORAGE_SCHEMA_VERSION = "cw008_storage_v2"

#: 单条决策记录的 schema 版本。CW-008 **不改变**记录契约。
RECORD_SCHEMA_VERSION = "cw007_decision_record_v1"

#: 冻结的 SQLite ``user_version``。**不得**随版本升级静默改写。
USER_VERSION = 2

#: 唯一被允许迁移的上一版 storage schema 与其 ``user_version``。
#:
#: 只支持这一条单向边（``v1 -> v2``）：没有 v0、没有跳迁、没有降级，也不是通用迁移框架。
LEGACY_STORAGE_SCHEMA_VERSION = "cw007_storage_v1"
LEGACY_USER_VERSION = 1

#: 冻结的连接级 PRAGMA。
BUSY_TIMEOUT_MS = 2000
JOURNAL_MODE = "DELETE"
SYNCHRONOUS = "FULL"
FOREIGN_KEYS = "ON"

#: 唯一允许存放数据库的项目相对目录与扩展名。
STORAGE_DIR = "artifacts/runtime"
STORAGE_SUFFIX = ".sqlite3"

#: 默认数据库（相对项目根目录）。
DEFAULT_STORAGE_PATH = f"{STORAGE_DIR}/cw007_decisions.sqlite3"

#: 与 CW-005/CW-006 API 契约一致的固定字符串。
#:
#: 刻意在这里再声明一次，而不是 ``from ai_detector.service.schemas import ...``：
#: ``service/__init__.py`` 会导入应用工厂，而应用工厂依赖 ``storage``，因此
#: ``storage → service`` 会形成循环导入。为了不让两处漂移，测试里有一条断言
#: （``test_storage_config.py::test_frozen_strings_match_the_api_contract``）逐字比较它们。
SCORE_SEMANTICS = "uncalibrated_model_score"
THRESHOLD_POLICY = "validation_f1"

#: 路径中不得出现的疑似凭据片段。
_CREDENTIAL_MARKERS = ("password", "token", "cookie", "api_key", "apikey", "secret")


class StorageErrorCode:
    """存储加载/运行失败的原因分类（用于启动诊断与对外错误码映射）。

    只表达类别，**不含**路径、SQL、记录值或异常文本。
    """

    NOT_CONFIGURED = "storage_not_configured"
    PATH_UNSAFE = "storage_path_unsafe"
    UNAVAILABLE = "storage_unavailable"
    SCHEMA_MISMATCH = "storage_schema_mismatch"
    CORRUPT = "storage_corrupt"
    CONTRACT_VIOLATION = "storage_contract_violation"
    UNEXPECTED = "storage_unexpected_error"


class StorageLoadError(RuntimeError):
    """存储无法安全打开或使用。

    Raises 时**不得**附带绝对路径、SQL、记录值或文件内容片段；用 :attr:`code` 表达原因类别，
    细节由调用方以安全措辞记录（或干脆丢弃）。
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def assert_safe_storage_path(value: Any, label: str = "storage.path") -> str:
    """纯字符串校验：路径必须是 ``artifacts/runtime/<文件名>.sqlite3``。

    判定**平台无关**：无论进程跑在哪个平台，下面这些形状都被拒绝——

    - Windows 盘符绝对/相对路径（``C:\\x``、``D:/x``、``C:x``）；
    - UNC 共享与扩展长度前缀（``\\\\server\\share\\x``、``\\\\?\\C:\\x``）；
    - POSIX 绝对路径（``/etc/x``）与根相对路径（``\\x``）；
    - 任何 ``..`` 组件；
    - 疑似凭据片段；
    - 不在 ``artifacts/runtime/`` 下、含额外目录层级、或扩展名不是 ``.sqlite3``。

    Raises:
        StorageLoadError: 任一条件不满足（``storage_path_unsafe``）。
    """
    text = str(value)
    if not text.strip():
        raise StorageLoadError(StorageErrorCode.PATH_UNSAFE, f"{label} 不能为空")

    if (
        PureWindowsPath(text).is_absolute()
        or PurePosixPath(text).is_absolute()
        or bool(PureWindowsPath(text).drive)
        or (len(text) > 1 and text[1] == ":")
        or text.startswith(("/", "\\"))
    ):
        raise StorageLoadError(StorageErrorCode.PATH_UNSAFE, f"{label} 必须是项目相对路径")

    normalized = text.replace("\\", "/")
    segments = normalized.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        # 空段（重复/结尾分隔符）、`.` 与 `..` 一律拒绝：只接受唯一一种字面形状。
        raise StorageLoadError(StorageErrorCode.PATH_UNSAFE, f"{label} 含非法路径组件")

    parts = PurePosixPath(normalized).parts
    lowered = text.lower()
    for marker in _CREDENTIAL_MARKERS:
        if marker in lowered:
            raise StorageLoadError(
                StorageErrorCode.PATH_UNSAFE, f"{label} 含疑似凭据字段 {marker}"
            )

    expected_dir = PurePosixPath(STORAGE_DIR).parts
    if len(parts) != len(expected_dir) + 1 or parts[: len(expected_dir)] != expected_dir:
        raise StorageLoadError(
            StorageErrorCode.PATH_UNSAFE,
            f"{label} 必须位于 {STORAGE_DIR}/ 下且只能是该目录的直接文件",
        )
    filename = parts[-1]
    if not filename.endswith(STORAGE_SUFFIX) or filename == STORAGE_SUFFIX:
        raise StorageLoadError(
            StorageErrorCode.PATH_UNSAFE, f"{label} 的扩展名必须是 {STORAGE_SUFFIX}"
        )
    return text


def resolve_real_path(path: Path) -> Path:
    """把路径解析为**真实路径**（解析符号链接/重解析点）。

    单独成函数是为了给测试一个**窄**的注入点：逃逸判定可以在不依赖"平台是否允许创建
    符号链接"的情况下被确定性地覆盖。
    """
    return Path(os.path.realpath(path))


def resolve_storage_path(value: Any) -> Path:
    """把安全的项目相对路径解析为绝对路径，并拒绝**符号链接/重解析点逃逸**。

    这是唯一做 I/O 的一步（``realpath``），因此只允许在 app lifespan 的存储初始化阶段调用，
    **不得**在配置解析或 ``create_app()`` 中调用。

    Raises:
        StorageLoadError: 路径形状不安全，或经链接解析后落到项目根之外。
    """
    relative = assert_safe_storage_path(value)
    candidate = resolve_path(relative)
    real_root = resolve_real_path(PROJECT_ROOT)
    real_candidate = resolve_real_path(candidate)
    if real_candidate != real_root and real_root not in real_candidate.parents:
        # 例如 ``artifacts/runtime`` 被替换成指向项目外的符号链接/联接点。
        raise StorageLoadError(
            StorageErrorCode.PATH_UNSAFE, "storage.path 经符号链接解析后逃出项目根目录"
        )
    return candidate
