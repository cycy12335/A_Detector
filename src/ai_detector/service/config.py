"""CW-005 服务配置：路径安全约束与启动参数。

本模块**只做配置解析**，不读取 bundle、不加载模型、不启动服务器——这些动作分别属于
:mod:`ai_detector.service.runtime` 与应用工厂。

## 路径纪律

配置里的每一个路径都必须是**相对项目根目录**的路径。绝对路径、盘符路径、``..`` 逃逸与
疑似凭据（``token``/``secret`` 等）一律拒绝，这样"把配置带到另一台机器"不会悄悄指向
本机任意位置，也不会把密钥写进版本库。

与 CW-004 的 ``_assert_project_relative`` 是**同一套规则的两个副本**：服务层刻意不依赖
训练模块，避免"只读服务"为了解析配置而把训练依赖整条拉进来。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ai_detector.config import PROJECT_ROOT, resolve_path
from ai_detector.rules.config import POLICY_SCHEMA_VERSION
from ai_detector.storage.config import (
    STORAGE_SCHEMA_VERSION,
    StorageLoadError,
    assert_safe_storage_path,
)

DEFAULT_CONFIG_PATH = "configs/service.yaml"

#: 路径中不得出现的疑似凭据片段。
_CREDENTIAL_MARKERS = ("password", "token", "cookie", "api_key", "apikey", "secret")

# ---------------------------------------------------------------------------
# 硬不变量：**写在代码里，配置文件不得放宽**
#
# 这些值决定了对外承诺的安全契约（批量上限、阈值策略、请求 ID 字符集、分数语义）。
# 如果只靠仓库 YAML 约束，那么改一行 YAML、或用 ``dataclasses.replace`` 构造一个
# 变体配置，就能让"最多 100 笔""固定 validation_f1""请求 ID 只允许固定字符集"
# 这些声明失效。因此它们在这里定义，并在 ``ServiceConfig.__post_init__`` 中校验——
# 该钩子在**任何**构造路径上都会执行，包括 ``from_mapping``、``from_yaml`` 与
# ``dataclasses.replace``。
# ---------------------------------------------------------------------------

#: 批量评分的**硬上限**。配置只能把它调得更小，不能调大。
HARD_BATCH_LIMIT = 100

#: 分数语义的固定字符串：未校准的模型分数。
FIXED_SCORE_SEMANTICS = "uncalibrated_model_score"

#: 阈值策略的固定字符串：冻结阈值来自 validation 上选出的 F1 阈值。
FIXED_THRESHOLD_POLICY = "validation_f1"

#: 请求 ID 的固定长度与字符集。配置**不得**放宽或替换。
REQUEST_ID_MAX_LENGTH = 64
REQUEST_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"

#: 允许绑定的回环地址。
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class ServiceConfigError(ValueError):
    """服务配置缺失或字段不合法。"""


def _assert_project_relative(value: Any, label: str) -> str:
    """硬校验：路径必须是项目相对路径。

    Raises:
        ServiceConfigError: 绝对路径、盘符路径、越出根目录或疑似凭据。
    """
    text = str(value)
    lowered = text.lower()
    if Path(text).is_absolute() or (len(text) > 1 and text[1] == ":"):
        raise ServiceConfigError(f"{label} 必须是项目相对路径，不得写绝对路径：{text}")
    if text.startswith(("/", "\\")) or ".." in Path(text).parts:
        raise ServiceConfigError(f"{label} 不得逃出项目根目录：{text}")
    for marker in _CREDENTIAL_MARKERS:
        if marker in lowered:
            raise ServiceConfigError(f"{label} 含疑似凭据字段 {marker}：{text}")
    return text


@dataclass(frozen=True)
class ServiceConfig:
    """``configs/service.yaml`` 的类型化表示。"""

    name: str
    api_version: str
    model_version: str
    host: str
    port: int
    batch_limit: int
    request_id_header: str
    score_semantics: str
    threshold_policy: str
    expected_source_features: int
    bundle_path: Path
    bundle_relative_path: str
    bundle_sha256: str
    expected_schema_version: str
    max_request_id_length: int
    request_id_pattern: str
    config_path: Path
    config_sha256: str = ""
    # CW-006 / 政策段。整体可以缺省（缺省即"未配置政策"，应用仍启动、政策接口降级），
    # 但一旦出现就必须**三个字段齐全**——半个 policy 段是配置错误，不允许部分生效。
    policy_path: Path | None = None
    policy_relative_path: str | None = None
    policy_sha256: str | None = None
    expected_policy_schema_version: str | None = None
    # CW-007 / 存储段。同样可以整体缺省（缺省即"未配置存储"，应用仍启动、decisions 接口降级），
    # 但一旦出现就必须**两个字段齐全**。路径被限制在项目相对的
    # ``artifacts/runtime/*.sqlite3``（见 ai_detector.storage.config）。
    storage_path: Path | None = None
    storage_relative_path: str | None = None
    storage_schema_version: str | None = None

    def __post_init__(self) -> None:
        """硬不变量校验。

        放在 ``__post_init__`` 而不是 ``from_mapping`` 里，是刻意的：``__post_init__`` 在
        **任何**构造路径上都会执行，因此 ``dataclasses.replace(config, batch_limit=101)``
        或直接构造都会被拦下，而不只是"从 YAML 读进来的那一份"。

        Raises:
            ServiceConfigError: 任一硬不变量被破坏。
        """
        if self.host not in LOOPBACK_HOSTS:
            raise ServiceConfigError(
                f"host 只允许回环地址 {sorted(LOOPBACK_HOSTS)}，实际为 {self.host!r}；"
                "本服务没有认证与 TLS，不得监听对外地址。"
            )
        if not 1 <= int(self.port) <= 65535:
            raise ServiceConfigError(f"port 必须位于 1-65535，实际为 {self.port}")
        if not 1 <= int(self.batch_limit) <= HARD_BATCH_LIMIT:
            raise ServiceConfigError(
                f"batch_limit 必须位于 1-{HARD_BATCH_LIMIT}，实际为 {self.batch_limit}；"
                "批量上限是固定的安全契约，配置不得放宽。"
            )
        if self.threshold_policy != FIXED_THRESHOLD_POLICY:
            raise ServiceConfigError(
                f"threshold_policy 必须是固定值 {FIXED_THRESHOLD_POLICY!r}，"
                f"实际为 {self.threshold_policy!r}；不得把它声明成其它策略名称。"
            )
        if self.score_semantics != FIXED_SCORE_SEMANTICS:
            raise ServiceConfigError(
                f"score_semantics 必须是固定值 {FIXED_SCORE_SEMANTICS!r}，"
                f"实际为 {self.score_semantics!r}；不得声明为概率或其它语义。"
            )
        if int(self.max_request_id_length) != REQUEST_ID_MAX_LENGTH:
            raise ServiceConfigError(
                f"max_request_id_length 必须是固定值 {REQUEST_ID_MAX_LENGTH}，"
                f"实际为 {self.max_request_id_length}"
            )
        if self.request_id_pattern != REQUEST_ID_PATTERN:
            raise ServiceConfigError(
                f"request_id_pattern 必须是固定值 {REQUEST_ID_PATTERN!r}，"
                f"实际为 {self.request_id_pattern!r}；请求 ID 的字符规则不得被配置放宽或替换。"
            )
        if int(self.expected_source_features) < 1:
            raise ServiceConfigError("expected_source_features 必须为正整数")
        self._validate_policy_section()
        self._validate_storage_section()

    def _validate_storage_section(self) -> None:
        """存储段的硬不变量：要么整体缺省，要么两个字段齐全且取值合法。

        路径只做**纯字符串**校验（平台无关的形状检查），因此配置解析**不做文件 I/O**；
        符号链接/重解析点逃逸检查发生在 app lifespan 的存储初始化阶段
        （``ai_detector.storage.config.resolve_storage_path``）。
        """
        declared = (self.storage_relative_path, self.storage_schema_version)
        if all(value is None for value in declared):
            if self.storage_path is not None:
                raise ServiceConfigError("storage 段未声明路径，但 storage_path 已被设置")
            return
        if any(value is None for value in declared):
            raise ServiceConfigError("storage 段缺少字段：path / schema_version")
        if self.storage_path is None:
            raise ServiceConfigError("storage 段声明了相对路径，但 storage_path 未被解析")
        if str(self.storage_schema_version) != STORAGE_SCHEMA_VERSION:
            raise ServiceConfigError(
                f"storage.schema_version 必须是固定值 {STORAGE_SCHEMA_VERSION!r}，"
                f"实际为 {self.storage_schema_version!r}"
            )
        try:
            assert_safe_storage_path(self.storage_relative_path)
        except StorageLoadError as error:
            # 存储层的错误分类不外泄给调用方；配置层只报"路径不安全"这一事实。
            raise ServiceConfigError(f"storage.path 不安全：{error.detail}") from error

    def _validate_policy_section(self) -> None:
        """政策段的硬不变量：要么整体缺省，要么三个字段齐全且取值合法。

        schema 版本必须等于代码里的 :data:`~ai_detector.rules.config.POLICY_SCHEMA_VERSION`：
        政策语义的版本号不允许由配置自由声明，否则"冻结政策"就只是句口号。
        """
        declared = (
            self.policy_relative_path,
            self.policy_sha256,
            self.expected_policy_schema_version,
        )
        if all(value is None for value in declared):
            if self.policy_path is not None:
                raise ServiceConfigError("policy 段未声明路径，但 policy_path 已被设置")
            return
        if any(value is None for value in declared):
            raise ServiceConfigError("policy 段缺少字段：path / sha256 / expected_schema_version")
        if self.policy_path is None:
            raise ServiceConfigError("policy 段声明了相对路径，但 policy_path 未被解析")
        if not re.fullmatch(r"[0-9a-f]{64}", str(self.policy_sha256)):
            raise ServiceConfigError("policy.sha256 必须是 64 位十六进制字符串")
        if str(self.expected_policy_schema_version) != POLICY_SCHEMA_VERSION:
            raise ServiceConfigError(
                f"policy.expected_schema_version 必须是固定值 {POLICY_SCHEMA_VERSION!r}，"
                f"实际为 {self.expected_policy_schema_version!r}"
            )

    def request_id_regex(self) -> re.Pattern[str]:
        """请求 ID 的校验式。

        **始终编译模块常量**，而不是配置里的字符串：即使有人绕过校验拿到一个被改过的
        config 对象，应用的行为也不会跟着放宽。
        """
        return re.compile(REQUEST_ID_PATTERN)

    # --- 加载 -----------------------------------------------------------

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        config_path: Path,
        config_sha256: str = "",
        allow_external_paths: bool = False,
    ) -> ServiceConfig:
        """从已解析的 YAML 映射构造配置。

        Args:
            allow_external_paths: 仅测试夹具使用。为 ``False``（仓库配置与命令行入口的默认值）时，
                ``bundle.path`` 必须是**项目相对路径**；测试夹具位于临时目录，必须显式打开该
                开关，因此"仓库配置里不得出现绝对路径"这条纪律无法被悄悄绕过。

        Raises:
            ServiceConfigError: 缺字段、类型不符或路径不安全。
        """
        for section in ("service", "bundle", "limits"):
            if section not in raw or not isinstance(raw[section], dict):
                raise ServiceConfigError(f"配置缺少 {section} 段")

        service = raw["service"]
        bundle = raw["bundle"]
        limits = raw["limits"]

        required_service = (
            "name",
            "api_version",
            "model_version",
            "host",
            "port",
            "batch_limit",
            "request_id_header",
            "score_semantics",
            "threshold_policy",
            "expected_source_features",
        )
        missing = [key for key in required_service if key not in service]
        if missing:
            raise ServiceConfigError("service 段缺少字段：" + ", ".join(missing))
        for key in ("path", "sha256", "expected_schema_version"):
            if key not in bundle:
                raise ServiceConfigError(f"bundle 段缺少字段：{key}")

        # policy 段**可缺省**：缺省表示"未配置政策"，应用仍可启动，只是政策相关能力降级。
        # 但一旦出现，三个字段必须齐全——半个政策段是配置错误，不允许部分生效。
        policy = raw.get("policy")
        policy_path: Path | None = None
        policy_relative: str | None = None
        policy_sha256: str | None = None
        policy_schema_version: str | None = None
        if policy is not None:
            if not isinstance(policy, dict):
                raise ServiceConfigError("policy 段必须是映射")
            missing_policy = [
                key
                for key in ("path", "sha256", "expected_schema_version")
                if key not in policy
            ]
            if missing_policy:
                raise ServiceConfigError("policy 段缺少字段：" + ", ".join(missing_policy))
            policy_sha256 = str(policy["sha256"]).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", policy_sha256):
                raise ServiceConfigError("policy.sha256 必须是 64 位十六进制字符串")
            policy_relative = (
                str(policy["path"])
                if allow_external_paths
                else _assert_project_relative(policy["path"], "policy.path")
            )
            policy_path = resolve_path(policy_relative)
            policy_schema_version = str(policy["expected_schema_version"])

        # storage 段同样可缺省；出现时两个字段必须齐全，且只允许这两个字段。
        storage = raw.get("storage")
        storage_path: Path | None = None
        storage_relative: str | None = None
        storage_schema_version: str | None = None
        if storage is not None:
            if not isinstance(storage, dict):
                raise ServiceConfigError("storage 段必须是映射")
            unexpected = sorted(set(storage) - {"path", "schema_version"})
            if unexpected:
                raise ServiceConfigError("storage 段含额外字段：" + ", ".join(unexpected))
            missing_storage = [
                key for key in ("path", "schema_version") if key not in storage
            ]
            if missing_storage:
                raise ServiceConfigError("storage 段缺少字段：" + ", ".join(missing_storage))
            storage_schema_version = str(storage["schema_version"])
            try:
                storage_relative = assert_safe_storage_path(storage["path"])
            except StorageLoadError as error:
                raise ServiceConfigError(f"storage.path 不安全：{error.detail}") from error
            storage_path = resolve_path(storage_relative)

        # 类型与格式解析；**硬不变量**统一由 __post_init__ 校验，
        # 这样 from_mapping / from_yaml / dataclasses.replace 三条路径行为一致。
        host = str(service["host"])
        port = int(service["port"])
        batch_limit = int(service["batch_limit"])
        expected_features = int(service["expected_source_features"])

        sha256 = str(bundle["sha256"]).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ServiceConfigError("bundle.sha256 必须是 64 位十六进制字符串")

        header = str(service["request_id_header"]).strip()
        if not header or not re.fullmatch(r"[A-Za-z0-9-]{1,64}", header):
            raise ServiceConfigError(f"request_id_header 不合法：{header!r}")

        max_len = int(limits.get("max_request_id_length", REQUEST_ID_MAX_LENGTH))
        pattern = str(limits.get("request_id_pattern", REQUEST_ID_PATTERN))
        # 正则必须是**可编译**的；是否等于固定值由 __post_init__ 判定。
        try:
            re.compile(pattern)
        except re.error as error:
            raise ServiceConfigError(f"limits.request_id_pattern 不是合法正则：{error}") from error

        relative = (
            str(bundle["path"])
            if allow_external_paths
            else _assert_project_relative(bundle["path"], "bundle.path")
        )

        return cls(
            name=str(service["name"]),
            api_version=str(service["api_version"]),
            model_version=str(service["model_version"]),
            host=host,
            port=port,
            batch_limit=batch_limit,
            request_id_header=header,
            score_semantics=str(service["score_semantics"]),
            threshold_policy=str(service["threshold_policy"]),
            expected_source_features=expected_features,
            bundle_path=resolve_path(relative),
            bundle_relative_path=relative,
            bundle_sha256=sha256,
            expected_schema_version=str(bundle["expected_schema_version"]),
            max_request_id_length=max_len,
            request_id_pattern=pattern,
            config_path=config_path,
            config_sha256=config_sha256,
            policy_path=policy_path,
            policy_relative_path=policy_relative,
            policy_sha256=policy_sha256,
            expected_policy_schema_version=policy_schema_version,
            storage_path=storage_path,
            storage_relative_path=storage_relative,
            storage_schema_version=storage_schema_version,
        )

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> ServiceConfig:
        """从 YAML 文件加载配置；默认读取 ``configs/service.yaml``。"""
        import hashlib

        config_path = resolve_path(path or DEFAULT_CONFIG_PATH)
        if not config_path.is_file():
            raise ServiceConfigError(f"服务配置文件不存在：{config_path}")
        raw_bytes = config_path.read_bytes()
        raw = yaml.safe_load(raw_bytes.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ServiceConfigError(f"服务配置顶层必须是映射：{config_path}")
        return cls.from_mapping(
            raw,
            config_path=config_path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    # --- 报告用 ---------------------------------------------------------

    def relative(self, path: str | Path) -> str:
        """报告用路径：根目录内给相对路径，否则退化为仅文件名（绝不写绝对路径）。"""
        from ai_detector.data.fingerprint import report_path

        return report_path(path, PROJECT_ROOT)[0]

    def public_dict(self) -> dict[str, Any]:
        """可安全写入报告/响应的配置摘要（不含绝对路径）。"""
        return {
            "service_name": self.name,
            "api_version": self.api_version,
            "model_version": self.model_version,
            "host": self.host,
            "port": self.port,
            "batch_limit": self.batch_limit,
            "request_id_header": self.request_id_header,
            "score_semantics": self.score_semantics,
            "threshold_policy": self.threshold_policy,
            "expected_source_features": self.expected_source_features,
            "bundle_relative_path": self.bundle_relative_path,
            "bundle_sha256": self.bundle_sha256,
            "config_relative_path": self.relative(self.config_path),
            "config_sha256": self.config_sha256,
            **self.policy_public_dict(),
            **self.storage_public_dict(),
        }

    def storage_public_dict(self) -> dict[str, Any]:
        """存储段的可安全公开摘要（不含绝对路径、SQL 或数据库内容）。"""
        return {
            "storage_configured": self.storage_path is not None,
            "storage_relative_path": (
                None if self.storage_path is None else self.relative(self.storage_path)
            ),
            "storage_schema_version": self.storage_schema_version,
        }

    def policy_public_dict(self) -> dict[str, Any]:
        """政策段的可安全公开摘要。

        ``policy_configured`` 明确区分"未配置政策"与"配置了但加载失败"——这两种情况
        在对外行为上都是政策接口 503，但在排障与验收时需要分得清。
        """
        return {
            "policy_configured": self.policy_path is not None,
            "policy_relative_path": (
                None if self.policy_path is None else self.relative(self.policy_path)
            ),
            "policy_sha256": self.policy_sha256,
            "policy_schema_version": self.expected_policy_schema_version,
        }
