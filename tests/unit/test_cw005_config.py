"""CW-005 服务配置的单元测试。

断言对象既包括**仓库里的正式配置**（这样"悄悄改成绝对路径或对外地址"会失败），
也包括配置解析本身的安全约束。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

# 服务专项依赖来自 `service` extra：未安装时清晰跳过本模块，安装后必须正常执行（不得跳过）。
pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from ai_detector.config import PROJECT_ROOT
from ai_detector.service.config import ServiceConfig, ServiceConfigError
from tests.cw005_fixtures import make_config

CONFIG_PATH = PROJECT_ROOT / "configs" / "service.yaml"
BUNDLE_PATH = PROJECT_ROOT / "artifacts/models/cw004/lightgbm_bundle.joblib"
BUNDLE_SHA256 = "f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c"


@pytest.fixture(scope="module")
def raw_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def config() -> ServiceConfig:
    return ServiceConfig.from_yaml(CONFIG_PATH)


# --- 正式配置 -----------------------------------------------------------


def test_repository_config_declares_required_sections(raw_config) -> None:
    for section in ("service", "bundle", "limits"):
        assert section in raw_config, section
    service = raw_config["service"]
    assert service["name"] == "ai-detector-risk-scoring"
    assert service["api_version"] == "v1"
    assert service["host"] == "127.0.0.1"
    assert service["port"] == 8000
    assert service["batch_limit"] == 100
    assert service["request_id_header"] == "X-Request-ID"
    assert service["score_semantics"] == "uncalibrated_model_score"
    assert service["threshold_policy"] == "validation_f1"
    assert service["expected_source_features"] == 199


def test_repository_config_pins_the_accepted_bundle(raw_config) -> None:
    bundle = raw_config["bundle"]
    assert bundle["path"] == "artifacts/models/cw004/lightgbm_bundle.joblib"
    assert bundle["sha256"] == BUNDLE_SHA256
    assert bundle["expected_schema_version"] == "cw004_bundle_v1"


def test_repository_config_pins_the_real_bundle_file() -> None:
    """配置里的指纹必须与磁盘上的正式 bundle 一致（否则服务永远起不来）。"""
    if not BUNDLE_PATH.is_file():  # pragma: no cover - 正式产物被清理时跳过
        pytest.skip("正式 bundle 不在位")
    digest = hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()
    assert digest == BUNDLE_SHA256


def test_repository_config_has_no_absolute_paths_or_credentials() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for marker in ("C:\\", "D:\\", "/Users/", "/home/"):
        assert marker not in text, marker
    for credential in ("password", "token", "cookie", "api_key", "secret"):
        assert credential not in text.lower(), credential
    raw = yaml.safe_load(text)
    assert not Path(raw["bundle"]["path"]).is_absolute()


def test_repository_config_loads_and_resolves_against_project_root(config: ServiceConfig) -> None:
    assert config.bundle_path.is_absolute()
    assert PROJECT_ROOT in config.bundle_path.parents
    assert config.config_sha256 == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert config.request_id_regex().fullmatch("abc-123")
    assert not config.request_id_regex().fullmatch("bad id with spaces")


def test_public_dict_exposes_no_absolute_path(config: ServiceConfig) -> None:
    payload = config.public_dict()
    assert payload["bundle_relative_path"] == "artifacts/models/cw004/lightgbm_bundle.joblib"
    assert payload["config_relative_path"] == "configs/service.yaml"
    for value in payload.values():
        assert not (isinstance(value, str) and (":\\" in value or value.startswith("/")))


# --- 配置解析的安全约束 -------------------------------------------------


def _base_mapping() -> dict:
    return {
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


def _from(raw: dict) -> ServiceConfig:
    return ServiceConfig.from_mapping(raw, config_path=Path("probe.yaml"))


def test_absolute_bundle_path_is_rejected() -> None:
    raw = _base_mapping()
    raw["bundle"]["path"] = "C:\\elsewhere\\bundle.joblib"
    with pytest.raises(ServiceConfigError, match="项目相对路径"):
        _from(raw)


def test_path_escaping_project_root_is_rejected() -> None:
    raw = _base_mapping()
    raw["bundle"]["path"] = "../outside/bundle.joblib"
    with pytest.raises(ServiceConfigError, match="逃出项目根目录"):
        _from(raw)


def test_credential_like_path_is_rejected() -> None:
    raw = _base_mapping()
    raw["bundle"]["path"] = "artifacts/models/token_bundle.joblib"
    with pytest.raises(ServiceConfigError, match="凭据"):
        _from(raw)


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "example.com", "192.168.1.10"])
def test_non_loopback_host_is_rejected(host: str) -> None:
    raw = _base_mapping()
    raw["service"]["host"] = host
    with pytest.raises(ServiceConfigError, match="回环地址"):
        _from(raw)


@pytest.mark.parametrize("port", [0, -1, 70000])
def test_invalid_port_is_rejected(port: int) -> None:
    raw = _base_mapping()
    raw["service"]["port"] = port
    with pytest.raises(ServiceConfigError, match="port"):
        _from(raw)


@pytest.mark.parametrize("value", ["not-hex", "a" * 63, "g" * 64, ""])
def test_invalid_bundle_sha256_is_rejected(value: str) -> None:
    raw = _base_mapping()
    raw["bundle"]["sha256"] = value
    with pytest.raises(ServiceConfigError, match="sha256"):
        _from(raw)


@pytest.mark.parametrize("header", ["", "bad header", "X" * 65])
def test_invalid_request_id_header_is_rejected(header: str) -> None:
    raw = _base_mapping()
    raw["service"]["request_id_header"] = header
    with pytest.raises(ServiceConfigError, match="request_id_header"):
        _from(raw)


@pytest.mark.parametrize("limit", [0, -5])
def test_invalid_batch_limit_is_rejected(limit: int) -> None:
    raw = _base_mapping()
    raw["service"]["batch_limit"] = limit
    with pytest.raises(ServiceConfigError, match="batch_limit"):
        _from(raw)


def test_missing_section_is_rejected() -> None:
    raw = _base_mapping()
    raw.pop("bundle")
    with pytest.raises(ServiceConfigError, match="缺少 bundle 段"):
        _from(raw)


def test_missing_service_field_is_rejected() -> None:
    raw = _base_mapping()
    raw["service"].pop("threshold_policy")
    with pytest.raises(ServiceConfigError, match="service 段缺少字段"):
        _from(raw)


def test_missing_config_file_raises() -> None:
    with pytest.raises(ServiceConfigError, match="不存在"):
        ServiceConfig.from_yaml("configs/does_not_exist.yaml")


def test_fixture_config_is_valid(tmp_path: Path) -> None:
    """夹具构造器本身也要产出一个合法配置（否则所有服务测试都建立在坏配置上）。"""
    config = make_config(tmp_path, features=7)
    assert config.expected_source_features == 7
    assert config.batch_limit == 100
    assert config.bundle_path.is_absolute()
