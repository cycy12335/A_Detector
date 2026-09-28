"""CW-006 政策配置的单元测试：路径安全、**先验哈希后解析**、严格 schema。

三条最要紧的性质：

1. **哈希不符时 ``yaml.safe_load`` 一次都不被调用**——顺序本身就是安全边界，
   与 CW-005 的 "先验哈希后 ``joblib.load``" 同源；
2. **配置不得放宽冻结语义**：额外字段、未知 operator、重复规则、非有限阈值、错误版本
   一律在加载阶段被拒，而不是"读进来再忽略"；
3. **仓库政策与代码常量一致**，因此改一行 YAML 既过不了哈希、也过不了硬不变量。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from ai_detector.config import PROJECT_ROOT
from ai_detector.rules.config import (
    FROZEN_BANDS,
    FROZEN_RULES,
    POLICY_SCHEMA_VERSION,
    POLICY_SEMANTICS,
    POLICY_VERSION,
    THRESHOLD_SOURCE,
    PolicyLoadError,
    PolicyErrorCode,
    load_policy,
)
from ai_detector.service.config import ServiceConfig, ServiceConfigError
from ai_detector.service.runtime import load_policy_engine
from tests.cw006_fixtures import (
    CANONICAL_POLICY,
    REPO_POLICY_PATH,
    policy_mapping,
    sha256_of,
    stage_dir,
    stage_policy,
    stage_relative,
)

SERVICE_CONFIG_PATH = PROJECT_ROOT / "configs" / "service.yaml"


@pytest.fixture(scope="module")
def repo_policy_raw() -> dict:
    return yaml.safe_load(REPO_POLICY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def service_raw() -> dict:
    return yaml.safe_load(SERVICE_CONFIG_PATH.read_text(encoding="utf-8"))


# --- 仓库产物 -----------------------------------------------------------


def test_repository_policy_is_pinned_in_the_service_config(
    repo_policy_raw: dict, service_raw: dict
) -> None:
    """服务配置里记录的路径与指纹必须正好指向磁盘上那份政策。"""
    assert service_raw["policy"]["path"] == "configs/risk_policy.yaml"
    assert service_raw["policy"]["expected_schema_version"] == POLICY_SCHEMA_VERSION
    assert service_raw["policy"]["sha256"] == sha256_of(REPO_POLICY_PATH)
    assert repo_policy_raw["policy"]["schema_version"] == POLICY_SCHEMA_VERSION


def test_repository_policy_matches_the_frozen_constants(repo_policy_raw: dict) -> None:
    """政策文件声明的每一条都必须与代码里的硬不变量逐字一致。"""
    section = repo_policy_raw["policy"]
    assert section["policy_version"] == POLICY_VERSION
    assert section["semantics"] == POLICY_SEMANTICS
    assert section["threshold_source"] == THRESHOLD_SOURCE
    for declared, frozen in zip(section["rules"], FROZEN_RULES):
        assert declared["rule_id"] == frozen.rule_id
        assert declared["feature"] == frozen.feature
        assert declared["operator"] == frozen.operator
        assert float(declared["threshold"]) == frozen.threshold
    assert len(section["rules"]) == len(FROZEN_RULES)
    declared_bands = {
        name: tuple(int(value) for value in span)
        for name, span in section["fusion"]["bands"].items()
    }
    assert declared_bands == {band.name: (band.low, band.high) for band in FROZEN_BANDS}


def test_repository_policy_matches_the_test_fixture_policy(repo_policy_raw: dict) -> None:
    """夹具政策不得悄悄漂移成一个与仓库不同的（更宽松的）政策。"""
    repo = repo_policy_raw["policy"]
    fixture = CANONICAL_POLICY["policy"]
    for key in ("schema_version", "policy_version", "semantics", "threshold_source", "rules", "fusion"):
        assert repo[key] == fixture[key], key
    # 免责声明允许措辞不同，但必须是一段真实的、说明"不是业务规则/概率"的声明。
    assert len(repo["disclaimer"]) >= 80
    for required in ("不是", "训练分区"):
        assert required in repo["disclaimer"], required


def test_repository_policy_has_no_absolute_paths_or_credentials() -> None:
    text = REPO_POLICY_PATH.read_text(encoding="utf-8")
    for marker in ("C:\\", "D:\\", "/Users/", "/home/"):
        assert marker not in text, marker
    for credential in ("password", "cookie", "api_key", "apikey", "secret"):
        assert credential not in text.lower(), credential


def test_repository_service_config_loads_the_policy_section() -> None:
    config = ServiceConfig.from_yaml(SERVICE_CONFIG_PATH)
    assert config.policy_relative_path == "configs/risk_policy.yaml"
    assert config.policy_path is not None and config.policy_path.is_absolute()
    assert config.policy_sha256 == sha256_of(REPO_POLICY_PATH)
    assert config.expected_policy_schema_version == POLICY_SCHEMA_VERSION
    public = config.public_dict()
    assert public["policy_relative_path"] == "configs/risk_policy.yaml"
    assert public["policy_sha256"] == sha256_of(REPO_POLICY_PATH)


def test_repository_policy_loads_into_a_policy_object() -> None:
    config = ServiceConfig.from_yaml(SERVICE_CONFIG_PATH)
    policy, digest = load_policy(
        config.policy_relative_path or "",
        expected_sha256=config.policy_sha256 or "",
        expected_schema_version=config.expected_policy_schema_version or "",
    )
    assert digest == sha256_of(REPO_POLICY_PATH)
    assert policy.policy_version == POLICY_VERSION
    assert [rule.rule_id for rule in policy.rules] == [rule.rule_id for rule in FROZEN_RULES]
    # 阈值与 CW-003 透明规则状态逐项一致（同一份 train-only q0.99 阈值）。
    cw003 = yaml.safe_load(
        (PROJECT_ROOT / "artifacts/models/cw003/rule_baseline_state.json").read_text(encoding="utf-8")
    )
    for rule in policy.rules:
        assert rule.threshold == float(cw003["thresholds"][rule.feature])


# --- 先验哈希，后解析 ---------------------------------------------------


def test_loader_rejects_an_existing_absolute_path(tmp_path: Path) -> None:
    """验收缺陷复现：绝对路径若真实存在且指纹匹配，不得被加载。

    这是首个独立验收的阻塞项 3：``load_policy()`` 曾经对绝对路径**跳过**路径校验，
    因此直接调用加载函数就能用项目外的一份政策成功加载，与"政策路径必须项目相对"
    的公开安全契约冲突。修复后路径校验无条件先执行，且**先于任何文件 I/O**。
    """
    policy_file, _relative = stage_policy(tmp_path)
    digest = sha256_of(policy_file)
    assert Path(str(policy_file)).is_absolute()

    with pytest.raises(PolicyLoadError) as error:
        load_policy(
            str(policy_file),
            expected_sha256=digest,
            expected_schema_version=POLICY_SCHEMA_VERSION,
        )
    assert error.value.code == PolicyErrorCode.PATH_UNSAFE


@pytest.mark.parametrize(
    "unsafe",
    [
        "C:\\elsewhere\\risk_policy.yaml",  # 盘符绝对路径
        "D:/elsewhere/risk_policy.yaml",  # 正斜杠盘符路径
        "\\\\server\\share\\risk_policy.yaml",  # UNC 共享
        "\\\\?\\C:\\elsewhere\\risk_policy.yaml",  # 扩展长度前缀
        "/etc/risk_policy.yaml",  # POSIX 绝对路径
        "\\risk_policy.yaml",  # 根相对
        "../outside/risk_policy.yaml",  # `..` 逃逸
        "configs/../../outside/risk_policy.yaml",  # 夹在中间的逃逸
        "artifacts/policy_token.yaml",  # 疑似凭据
    ],
)
def test_loader_rejects_unsafe_paths_before_any_io(tmp_path: Path, unsafe: str) -> None:
    """无论平台，路径不安全都必须先于文件存在性检查被拒绝。"""
    with pytest.raises(PolicyLoadError) as error:
        load_policy(
            unsafe,
            expected_sha256="a" * 64,
            expected_schema_version=POLICY_SCHEMA_VERSION,
        )
    assert error.value.code == PolicyErrorCode.PATH_UNSAFE


def test_loader_accepts_a_project_relative_path(tmp_path: Path) -> None:
    """生产入口的形状：项目相对路径 + 固定指纹，正常加载。"""
    policy_file, relative = stage_policy(tmp_path)
    assert not Path(relative).is_absolute()
    policy, digest = load_policy(
        relative,
        expected_sha256=sha256_of(policy_file),
        expected_schema_version=POLICY_SCHEMA_VERSION,
    )
    assert digest == sha256_of(policy_file)
    assert policy.schema_version == POLICY_SCHEMA_VERSION


def test_production_loader_rejects_an_absolute_policy_path_from_a_config(tmp_path: Path) -> None:
    """``load_policy_engine`` 是生产入口：即使配置对象谎报绝对路径也必须拒绝。

    用鸭子类型的配置替身绕过 ``ServiceConfig`` 的构造校验，验证"底层可绕过"这个
    验收阻塞项确实已被封死——生产入口**没有**任何放宽路径检查的开关。
    """
    policy_file, _relative = stage_policy(tmp_path)

    @dataclass
    class _StubConfig:
        policy_relative_path: str
        policy_sha256: str
        expected_policy_schema_version: str

    stub = _StubConfig(
        policy_relative_path=str(policy_file),
        policy_sha256=sha256_of(policy_file),
        expected_policy_schema_version=POLICY_SCHEMA_VERSION,
    )
    with pytest.raises(PolicyLoadError) as error:
        load_policy_engine(stub)
    assert error.value.code == PolicyErrorCode.PATH_UNSAFE


def test_production_loader_loads_a_project_relative_policy_from_a_config(tmp_path: Path) -> None:
    policy_file, relative = stage_policy(tmp_path)

    @dataclass
    class _StubConfig:
        policy_relative_path: str
        policy_sha256: str
        expected_policy_schema_version: str

    engine = load_policy_engine(
        _StubConfig(
            policy_relative_path=relative,
            policy_sha256=sha256_of(policy_file),
            expected_policy_schema_version=POLICY_SCHEMA_VERSION,
        )
    )
    assert engine.policy_version == POLICY_VERSION
    assert engine.policy_sha256 == sha256_of(policy_file)


def test_load_policy_is_not_called_when_the_hash_mismatches(tmp_path: Path, monkeypatch) -> None:
    """指纹不符时必须在 ``yaml.safe_load`` **之前**失败。"""
    _policy_file, relative = stage_policy(tmp_path)

    parses: list[object] = []
    real_load = yaml.safe_load

    def _recording_load(stream, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        parses.append(stream)
        return real_load(stream, *args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", _recording_load)

    with pytest.raises(PolicyLoadError) as error:
        load_policy(
            relative,
            expected_sha256="c" * 64,
            expected_schema_version=POLICY_SCHEMA_VERSION,
        )
    assert error.value.code == PolicyErrorCode.HASH_MISMATCH
    assert parses == [], "指纹校验失败后不得解析 YAML"


def test_missing_policy_file_fails_before_parsing(tmp_path: Path, monkeypatch) -> None:
    parses: list[object] = []
    monkeypatch.setattr(yaml, "safe_load", lambda stream, *a, **k: parses.append(stream))

    with pytest.raises(PolicyLoadError) as error:
        load_policy(
            stage_relative(tmp_path, "does_not_exist.yaml"),
            expected_sha256="d" * 64,
            expected_schema_version=POLICY_SCHEMA_VERSION,
        )
    assert error.value.code == PolicyErrorCode.MISSING
    assert parses == []


def test_load_error_detail_contains_no_path(tmp_path: Path) -> None:
    _policy_file, relative = stage_policy(tmp_path)
    with pytest.raises(PolicyLoadError) as error:
        load_policy(
            relative, expected_sha256="e" * 64, expected_schema_version=POLICY_SCHEMA_VERSION
        )
    assert str(tmp_path) not in error.value.detail
    assert "C:\\" not in error.value.detail
    assert "artifacts" not in error.value.detail


def test_valid_policy_loads_and_reports_the_digest(tmp_path: Path) -> None:
    policy_file, relative = stage_policy(tmp_path)
    policy, digest = load_policy(
        relative,
        expected_sha256=sha256_of(policy_file),
        expected_schema_version=POLICY_SCHEMA_VERSION,
    )
    assert digest == hashlib.sha256(policy_file.read_bytes()).hexdigest()
    assert policy.schema_version == POLICY_SCHEMA_VERSION


# --- 严格 schema：配置不得放宽冻结语义 ----------------------------------


def _load(tmp_path: Path, raw: dict, *, name: str = "p.yaml"):
    """写一份自定义政策并通过**生产加载函数**加载（走项目相对路径的真实链路）。"""
    policy_file, relative = stage_policy(tmp_path, raw, filename=name)
    return load_policy(
        relative,
        expected_sha256=sha256_of(policy_file),
        expected_schema_version=POLICY_SCHEMA_VERSION,
    )


def test_extra_top_level_key_is_rejected(tmp_path: Path) -> None:
    raw = policy_mapping()
    raw["extra_section"] = {}
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, raw)
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


@pytest.mark.parametrize(
    "key,value",
    [
        ("unexpected_flag", True),
        ("relaxed", {"any": "thing"}),
        ("notes", "自由文本"),
    ],
)
def test_extra_policy_field_is_rejected(tmp_path: Path, key: str, value: object) -> None:
    """多一个字段就拒绝——不允许"看起来被接受、实际被忽略"的配置。"""
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(**{key: value}))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


@pytest.mark.parametrize(
    "key",
    ["schema_version", "policy_version", "semantics", "threshold_source", "rules", "fusion", "disclaimer"],
)
def test_missing_required_field_is_rejected(tmp_path: Path, key: str) -> None:
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(**{key: None}))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", "cw006_policy_v2"),
        ("schema_version", "cw005_policy_v1"),
        ("policy_version", "cw006-risk-policy-v2"),
        ("policy_version", "cw006-risk-policy-v1 "),
        ("semantics", "decision_band"),
        ("semantics", "production_decision"),
        ("threshold_source", "business_rule"),
        ("threshold_source", "test_q0.99"),
    ],
)
def test_non_canonical_version_or_semantics_is_rejected(
    tmp_path: Path, key: str, value: str
) -> None:
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(**{key: value}))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


def test_wrong_schema_version_in_the_service_config_is_rejected(tmp_path: Path) -> None:
    """服务配置声明的政策 schema 版本也必须是冻结值。"""
    policy_file, _relative = stage_policy(tmp_path)
    raw = _service_mapping(
        policy={
            "path": str(policy_file),
            "sha256": sha256_of(policy_file),
            "expected_schema_version": "cw006_policy_v99",
        }
    )
    with pytest.raises(ServiceConfigError, match="schema"):
        ServiceConfig.from_mapping(raw, config_path=tmp_path / "service.yaml", allow_external_paths=True)


def test_rule_order_is_frozen(tmp_path: Path) -> None:
    rules = list(CANONICAL_POLICY["policy"]["rules"])
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(rules=[rules[1], rules[0]]))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


def test_extra_rule_is_rejected(tmp_path: Path) -> None:
    rules = list(CANONICAL_POLICY["policy"]["rules"])
    extra = {"rule_id": "third_rule", "feature": "D1", "operator": "gt", "threshold": 1.0}
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(rules=[*rules, extra]))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


def test_duplicate_rule_id_is_rejected(tmp_path: Path) -> None:
    rules = list(CANONICAL_POLICY["policy"]["rules"])
    cloned = dict(rules[0])
    cloned["feature"] = "V1"  # 保留第一个 rule_id，制造重复
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(rules=[rules[0], cloned]))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


def test_duplicate_feature_is_rejected(tmp_path: Path) -> None:
    rules = list(CANONICAL_POLICY["policy"]["rules"])
    cloned = dict(rules[1])
    cloned["feature"] = rules[0]["feature"]
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(rules=[rules[0], cloned]))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


@pytest.mark.parametrize("operator", ["gte", "ge", "GT", ">", "lt", "eq", "in", ""])
def test_unknown_operator_is_rejected(tmp_path: Path, operator: str) -> None:
    """只允许严格 ``gt``：``>=`` 会让"恰好等于阈值"也命中，冻结语义不容替换。"""
    rules = list(CANONICAL_POLICY["policy"]["rules"])
    patched = [dict(rules[0], operator=operator), dict(rules[1])]
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(rules=patched))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), "1092.0", None, True])
def test_non_finite_or_non_numeric_threshold_is_rejected(tmp_path: Path, threshold: object) -> None:
    rules = list(CANONICAL_POLICY["policy"]["rules"])
    patched = [dict(rules[0], threshold=threshold), dict(rules[1])]
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(rules=patched))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


@pytest.mark.parametrize("threshold", [1000.0, 1092.5, 0.0, -1.0])
def test_threshold_not_matching_the_cw003_value_is_rejected(
    tmp_path: Path, threshold: float
) -> None:
    """阈值只能等于 CW-003 train-only q0.99 的结果，不能"现场调参"。"""
    rules = list(CANONICAL_POLICY["policy"]["rules"])
    patched = [dict(rules[0], threshold=threshold), dict(rules[1])]
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(rules=patched))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


def test_extra_rule_field_is_rejected(tmp_path: Path) -> None:
    rules = list(CANONICAL_POLICY["policy"]["rules"])
    patched = [dict(rules[0], weight=3), dict(rules[1])]
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(rules=patched))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


@pytest.mark.parametrize(
    "fusion_patch",
    [
        {"flagged_signal_points": 3},
        {"matched_rule_signal_points": 2},
        {"rule_score_denominator": 4},
        {"flagged_signal_points": 2, "matched_rule_signal_points": 1, "rule_score_denominator": 2,
         "bands": {"low_signal": [0, 0], "elevated_signal": [1, 3], "high_signal": [4, 4]}},
        {"flagged_signal_points": 2, "matched_rule_signal_points": 1, "rule_score_denominator": 2,
         "bands": {"low_signal": [1, 1], "elevated_signal": [2, 2], "high_signal": [3, 4]}},
    ],
)
def test_loosened_fusion_table_is_rejected(tmp_path: Path, fusion_patch: dict) -> None:
    """融合真值表是冻结的：配置不得改点值或移动区间。"""
    fusion = dict(CANONICAL_POLICY["policy"]["fusion"])
    fusion.update(fusion_patch)
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(fusion=fusion))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


@pytest.mark.parametrize("disclaimer", ["", "   ", None, 5])
def test_missing_or_empty_disclaimer_is_rejected(tmp_path: Path, disclaimer: object) -> None:
    with pytest.raises(PolicyLoadError) as error:
        _load(tmp_path, policy_mapping(disclaimer=disclaimer))
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


def test_malformed_yaml_is_folded_into_a_safe_code(tmp_path: Path) -> None:
    broken = stage_dir(tmp_path) / "broken.yaml"
    broken.write_text("policy: [unclosed\n", encoding="utf-8")
    with pytest.raises(PolicyLoadError) as error:
        load_policy(
            broken.relative_to(PROJECT_ROOT).as_posix(),
            expected_sha256=sha256_of(broken),
            expected_schema_version=POLICY_SCHEMA_VERSION,
        )
    assert error.value.code == PolicyErrorCode.PARSE_FAILED
    assert str(tmp_path) not in error.value.detail
    assert "artifacts" not in error.value.detail


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    policy_file = stage_dir(tmp_path) / "list.yaml"
    policy_file.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(PolicyLoadError) as error:
        load_policy(
            policy_file.relative_to(PROJECT_ROOT).as_posix(),
            expected_sha256=sha256_of(policy_file),
            expected_schema_version=POLICY_SCHEMA_VERSION,
        )
    assert error.value.code == PolicyErrorCode.SCHEMA_VIOLATION


# --- 路径安全 -----------------------------------------------------------


def _service_mapping(**patches: dict) -> dict:
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
    raw.update(patches)
    return raw


@pytest.mark.parametrize(
    "path",
    [
        "C:\\elsewhere\\risk_policy.yaml",
        "/etc/risk_policy.yaml",
        "../outside/risk_policy.yaml",
        "configs/../../outside/risk_policy.yaml",
        "artifacts/policy_token.yaml",
    ],
)
def test_unsafe_policy_path_is_rejected_by_the_service_config(tmp_path: Path, path: str) -> None:
    raw = _service_mapping(
        policy={"path": path, "sha256": "a" * 64, "expected_schema_version": POLICY_SCHEMA_VERSION}
    )
    with pytest.raises(ServiceConfigError):
        ServiceConfig.from_mapping(raw, config_path=tmp_path / "service.yaml")


@pytest.mark.parametrize("sha", ["not-hex", "a" * 63, "g" * 64, ""])
def test_invalid_policy_sha256_is_rejected(tmp_path: Path, sha: str) -> None:
    raw = _service_mapping(
        policy={
            "path": "configs/risk_policy.yaml",
            "sha256": sha,
            "expected_schema_version": POLICY_SCHEMA_VERSION,
        }
    )
    with pytest.raises(ServiceConfigError, match="sha256"):
        ServiceConfig.from_mapping(raw, config_path=tmp_path / "service.yaml")


def test_incomplete_policy_section_is_rejected(tmp_path: Path) -> None:
    """只写一半的 policy 段是配置错误，不允许"部分生效"。"""
    raw = _service_mapping(policy={"path": "configs/risk_policy.yaml"})
    with pytest.raises(ServiceConfigError, match="policy 段缺少字段"):
        ServiceConfig.from_mapping(raw, config_path=tmp_path / "service.yaml")


def test_service_config_without_a_policy_section_is_accepted(tmp_path: Path) -> None:
    """"未配置政策"是合法状态：应用仍启动，只是政策相关接口降级。"""
    config = ServiceConfig.from_mapping(
        _service_mapping(), config_path=tmp_path / "service.yaml"
    )
    assert config.policy_path is None
    assert config.policy_sha256 is None
    assert config.policy_public_dict() == {
        "policy_configured": False,
        "policy_relative_path": None,
        "policy_sha256": None,
        "policy_schema_version": None,
    }
