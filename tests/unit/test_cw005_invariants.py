"""CW-005 硬不变量与启动诊断的定向测试。

首轮独立验收指出：批量上限、阈值策略、请求 ID 字符规则、分数语义这四条**对外承诺**
只写在仓库 YAML 里，因此改一行 YAML（或用 ``dataclasses.replace`` 造一个变体配置）
就能让承诺失效。本文件把每条不变量钉在**代码**上：

- 不变量在 ``ServiceConfig.__post_init__`` 里校验，该钩子在 ``from_mapping``、``from_yaml``
  与 ``dataclasses.replace`` **三条路径**上都会执行；
- 请求 ID 的校验式始终编译模块常量，即使配置字符串被换掉，应用行为也不放宽；
- ``load_runtime`` 保留对分数语义与阈值策略的二次检查（纵深防御）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml

from ai_detector.service.config import (
    FIXED_SCORE_SEMANTICS,
    FIXED_THRESHOLD_POLICY,
    HARD_BATCH_LIMIT,
    REQUEST_ID_MAX_LENGTH,
    REQUEST_ID_PATTERN,
    ServiceConfig,
    ServiceConfigError,
)
from ai_detector.service.errors import BundleErrorCode, BundleLoadError
from ai_detector.service.runtime import load_runtime
from tests.cw005_fixtures import make_config, sha256_of, write_synthetic_bundle


def _mapping(**overrides) -> dict:
    raw = {
        "service": {
            "name": "svc",
            "api_version": "v1",
            "model_version": "m1",
            "host": "127.0.0.1",
            "port": 8000,
            "batch_limit": 100,
            "request_id_header": "X-Request-ID",
            "score_semantics": FIXED_SCORE_SEMANTICS,
            "threshold_policy": FIXED_THRESHOLD_POLICY,
            "expected_source_features": 199,
        },
        "bundle": {
            "path": "artifacts/models/cw004/lightgbm_bundle.joblib",
            "sha256": "a" * 64,
            "expected_schema_version": "cw004_bundle_v1",
        },
        "limits": {
            "max_request_id_length": REQUEST_ID_MAX_LENGTH,
            "request_id_pattern": REQUEST_ID_PATTERN,
        },
    }
    for section, patch in overrides.items():
        raw[section].update(patch)
    return raw


def _from(raw: dict) -> ServiceConfig:
    return ServiceConfig.from_mapping(raw, config_path=Path("probe.yaml"))


# --- 硬不变量：写在代码里，配置不得放宽 ---------------------------------


@pytest.mark.parametrize("value", [101, 500, 10**6])
def test_batch_limit_above_the_hard_cap_is_rejected(value: int) -> None:
    with pytest.raises(ServiceConfigError, match="batch_limit"):
        _from(_mapping(service={"batch_limit": value}))


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_batch_limit_is_rejected(value: int) -> None:
    with pytest.raises(ServiceConfigError, match="batch_limit"):
        _from(_mapping(service={"batch_limit": value}))


@pytest.mark.parametrize("value", [1, 10, HARD_BATCH_LIMIT])
def test_batch_limit_at_or_below_the_cap_is_accepted(value: int) -> None:
    """收紧（调小）是允许的，放宽才是被拒绝的。"""
    config = _from(_mapping(service={"batch_limit": value}))
    assert config.batch_limit == value


@pytest.mark.parametrize("value", ["arbitrary_policy", "", "f1", "VALIDATION_F1", None])
def test_non_canonical_threshold_policy_is_rejected(value: object) -> None:
    with pytest.raises(ServiceConfigError, match="threshold_policy"):
        _from(_mapping(service={"threshold_policy": value}))


@pytest.mark.parametrize("value", ["fraud_probability", "probability", "", "score"])
def test_non_canonical_score_semantics_is_rejected(value: str) -> None:
    with pytest.raises(ServiceConfigError, match="score_semantics"):
        _from(_mapping(service={"score_semantics": value}))


@pytest.mark.parametrize("pattern", [r"^.*$", r"^.+$", r"^[A-Za-z0-9 ._:-]{1,64}$", r".*"])
def test_loosened_request_id_pattern_is_rejected(pattern: str) -> None:
    with pytest.raises(ServiceConfigError, match="request_id_pattern"):
        _from(_mapping(limits={"request_id_pattern": pattern}))


@pytest.mark.parametrize("length", [8, 128, 0])
def test_non_canonical_request_id_length_is_rejected(length: int) -> None:
    with pytest.raises(ServiceConfigError, match="max_request_id_length"):
        _from(_mapping(limits={"max_request_id_length": length}))


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.1", "example.com"])
def test_non_loopback_host_is_rejected(host: str) -> None:
    with pytest.raises(ServiceConfigError, match="回环地址"):
        _from(_mapping(service={"host": host}))


# --- 构造路径：dataclasses.replace 也必须被拦下 --------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_limit", 101),
        ("batch_limit", 0),
        ("threshold_policy", "arbitrary_policy"),
        ("score_semantics", "fraud_probability"),
        ("request_id_pattern", r"^.*$"),
        ("max_request_id_length", 8),
        ("host", "0.0.0.0"),
    ],
)
def test_replace_cannot_bypass_hard_invariants(tmp_path: Path, field: str, value: object) -> None:
    """``dataclasses.replace`` 会重新执行 ``__post_init__``，因此同样被拦下。

    这正是首轮验收用来绕过配置层的路径；不变量放在 ``__post_init__`` 之后它不再成立。
    """
    config = make_config(tmp_path)
    with pytest.raises(ServiceConfigError):
        replace(config, **{field: value})


def test_request_id_regex_is_always_the_module_constant(tmp_path: Path) -> None:
    """即使拿到一个被改过的 config 对象，应用的校验式也不放宽。"""

    config = make_config(tmp_path)
    assert config.request_id_regex().pattern == REQUEST_ID_PATTERN
    # 直接用对象替换（绕过 __post_init__ 的伪造方式）也不改变校验式。
    forged = object.__new__(ServiceConfig)
    object.__setattr__(forged, "request_id_pattern", r"^.*$")
    assert forged.request_id_regex().pattern == REQUEST_ID_PATTERN
    assert not forged.request_id_regex().fullmatch("bad/id with spaces")
    assert forged.request_id_regex().fullmatch("good-id_1.2:3")


# --- 运行时纵深防御 -----------------------------------------------------


@dataclass
class _StubConfig:
    """鸭子类型的配置替身：绕过 ``ServiceConfig`` 的构造校验，用于验证运行时自己的检查。"""

    bundle_path: Path
    bundle_sha256: str
    expected_schema_version: str
    expected_source_features: int
    score_semantics: str
    threshold_policy: str
    model_version: str


def _stub(tmp_path: Path, **overrides) -> _StubConfig:
    bundle = write_synthetic_bundle(tmp_path / "bundle.joblib", features=["a", "b", "c"])
    values = {
        "bundle_path": bundle,
        "bundle_sha256": sha256_of(bundle),
        "expected_schema_version": "cw004_bundle_v1",
        "expected_source_features": 3,
        "score_semantics": FIXED_SCORE_SEMANTICS,
        "threshold_policy": FIXED_THRESHOLD_POLICY,
        "model_version": "m1",
    }
    values.update(overrides)
    return _StubConfig(**values)


def test_runtime_rejects_a_non_canonical_threshold_policy(tmp_path: Path) -> None:
    """即使配置对象绕过了构造校验，``load_runtime`` 也不接受任意策略名。"""
    with pytest.raises(BundleLoadError) as error:
        load_runtime(_stub(tmp_path, threshold_policy="arbitrary_policy"))
    assert error.value.code == BundleErrorCode.SCHEMA_MISMATCH
    assert "threshold_policy" in error.value.detail


def test_runtime_rejects_a_non_canonical_score_semantics(tmp_path: Path) -> None:
    with pytest.raises(BundleLoadError) as error:
        load_runtime(_stub(tmp_path, score_semantics="fraud_probability"))
    assert error.value.code == BundleErrorCode.SCHEMA_MISMATCH


def test_runtime_accepts_the_canonical_stub(tmp_path: Path) -> None:
    runtime = load_runtime(_stub(tmp_path))
    assert runtime.threshold_policy == FIXED_THRESHOLD_POLICY
    assert runtime.score_semantics == FIXED_SCORE_SEMANTICS


# --- 仓库配置必须与代码常量一致 -----------------------------------------


def test_repository_config_matches_the_code_constants() -> None:
    from ai_detector.config import PROJECT_ROOT

    raw = yaml.safe_load((PROJECT_ROOT / "configs/service.yaml").read_text(encoding="utf-8"))
    assert raw["service"]["batch_limit"] == HARD_BATCH_LIMIT
    assert raw["service"]["threshold_policy"] == FIXED_THRESHOLD_POLICY
    assert raw["service"]["score_semantics"] == FIXED_SCORE_SEMANTICS
    assert raw["limits"]["request_id_pattern"] == REQUEST_ID_PATTERN
    assert raw["limits"]["max_request_id_length"] == REQUEST_ID_MAX_LENGTH
