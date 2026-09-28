"""CW-005 运行时加载的单元测试：**先验哈希，后反序列化**，再校验契约。

最要紧的一条：bundle 指纹不符时 ``joblib.load`` **一次都不能被调用**。
``joblib.load`` 等价于 ``pickle.load``，对不可信文件可以执行任意代码，因此顺序就是安全边界。
这里用 monkeypatch 记录调用并把这条性质钉死。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

# 服务专项依赖来自 `service` extra：未安装时清晰跳过本模块，安装后必须正常执行（不得跳过）。
pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from pydantic import ValidationError

from ai_detector.service.errors import BundleErrorCode, BundleLoadError
from ai_detector.service.runtime import (
    REQUIRED_SCORE_SEMANTICS,
    load_runtime,
    sha256_of_file,
    verify_bundle_hash,
)
from tests.cw005_fixtures import make_config, sha256_of, write_synthetic_bundle

FEATURES = ["feat_a", "feat_b", "feat_c"]


def _config_for(tmp_path: Path, bundle: Path, **overrides):
    return make_config(
        tmp_path,
        features=overrides.pop("features", len(FEATURES)),
        bundle_path=str(bundle),
        bundle_sha256=sha256_of(bundle),
        **overrides,
    )


@pytest.fixture
def bundle_path(tmp_path: Path) -> Path:
    return write_synthetic_bundle(tmp_path / "bundle.joblib", features=FEATURES)


# --- hash-before-load ---------------------------------------------------


def test_joblib_load_is_not_called_on_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    """指纹不符时必须在反序列化**之前**失败。"""
    bundle = write_synthetic_bundle(tmp_path / "bundle.joblib", features=FEATURES)
    config = make_config(
        tmp_path, features=len(FEATURES), bundle_path=str(bundle), bundle_sha256="b" * 64
    )

    calls: list[Path] = []
    real_load = joblib.load

    def _recording_load(path, *args, **kwargs):
        calls.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", _recording_load)

    with pytest.raises(BundleLoadError) as error:
        load_runtime(config)
    assert error.value.code == BundleErrorCode.HASH_MISMATCH
    assert calls == [], "指纹校验失败后不得调用 joblib.load"


def test_missing_bundle_fails_before_deserialization(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path, features=len(FEATURES), bundle_path="artifacts/nope.joblib")

    calls: list[Path] = []
    monkeypatch.setattr(joblib, "load", lambda path, *a, **k: calls.append(Path(path)))

    with pytest.raises(BundleLoadError) as error:
        load_runtime(config)
    assert error.value.code == BundleErrorCode.MISSING
    assert calls == []


def test_verify_bundle_hash_returns_the_digest(bundle_path: Path) -> None:
    digest = verify_bundle_hash(bundle_path, sha256_of(bundle_path))
    assert digest == sha256_of_file(bundle_path)
    assert len(digest) == 64


def test_load_error_detail_contains_no_path_or_digest_leak(bundle_path: Path) -> None:
    """错误说明里不得出现绝对路径。"""
    with pytest.raises(BundleLoadError) as error:
        verify_bundle_hash(bundle_path, "c" * 64)
    assert str(bundle_path.parent) not in error.value.detail
    assert "C:\\" not in error.value.detail


# --- 契约校验 -----------------------------------------------------------


def test_successful_load_exposes_scoring_runtime(tmp_path: Path, bundle_path: Path) -> None:
    runtime = load_runtime(_config_for(tmp_path, bundle_path))
    assert runtime.source_features == FEATURES
    assert runtime.score_semantics == REQUIRED_SCORE_SEMANTICS
    assert 0.0 <= runtime.threshold <= 1.0
    assert runtime.threshold_policy == "validation_f1"
    assert runtime.bundle_sha256 == sha256_of(bundle_path)
    assert runtime.numeric_features == FEATURES


def test_feature_count_mismatch_is_rejected(tmp_path: Path, bundle_path: Path) -> None:
    config = _config_for(tmp_path, bundle_path, features=len(FEATURES) + 1)
    with pytest.raises(BundleLoadError) as error:
        load_runtime(config)
    assert error.value.code == BundleErrorCode.FEATURE_CONTRACT


def test_duplicate_feature_names_are_rejected(tmp_path: Path) -> None:
    bundle = write_synthetic_bundle(
        tmp_path / "dup.joblib", features=["a", "a", "b"]
    )
    config = _config_for(tmp_path, bundle)
    with pytest.raises(BundleLoadError) as error:
        load_runtime(config)
    assert error.value.code == BundleErrorCode.FEATURE_CONTRACT


def test_schema_version_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = write_synthetic_bundle(tmp_path / "b.joblib", features=FEATURES)
    config = make_config(
        tmp_path,
        features=len(FEATURES),
        bundle_path=str(bundle),
        bundle_sha256=sha256_of(bundle),
        schema_version="cw004_bundle_v99",
    )
    with pytest.raises(BundleLoadError) as error:
        load_runtime(config)
    assert error.value.code == BundleErrorCode.SCHEMA_MISMATCH


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(tmp_path: Path, threshold: float) -> None:
    bundle = write_synthetic_bundle(tmp_path / "t.joblib", features=FEATURES, threshold=threshold)
    config = _config_for(tmp_path, bundle)
    with pytest.raises(BundleLoadError) as error:
        load_runtime(config)
    assert error.value.code == BundleErrorCode.THRESHOLD_INVALID


def test_wrong_payload_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not_a_bundle.joblib"
    joblib.dump({"not": "a bundle"}, path)
    with pytest.raises(BundleLoadError) as error:
        load_runtime(_config_for(tmp_path, path))
    assert error.value.code == BundleErrorCode.WRONG_TYPE


def test_deserialization_failure_is_folded_into_a_safe_code(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"this is not a pickle")
    with pytest.raises(BundleLoadError) as error:
        load_runtime(_config_for(tmp_path, path))
    assert error.value.code == BundleErrorCode.DESERIALIZE_FAILED


def test_bundle_marked_as_containing_raw_data_is_rejected(tmp_path: Path) -> None:
    """声明含原始数据的 bundle 必须被拒（服务是只读推理端，不该装数据）。"""
    from tests.cw005_fixtures import RawDataReportingBundle

    bundle = write_synthetic_bundle(
        tmp_path / "raw.joblib", features=FEATURES, bundle_class=RawDataReportingBundle
    )
    with pytest.raises(BundleLoadError) as error:
        load_runtime(_config_for(tmp_path, bundle))
    assert error.value.code == BundleErrorCode.CONTAINS_RAW_DATA


def test_score_semantics_cannot_be_tampered_with(tmp_path: Path, bundle_path: Path) -> None:
    """配置层已经不接受非规范分数语义；``dataclasses.replace`` 也绕不过去。

    运行时的二次检查（针对鸭子类型的伪造对象）见 ``test_cw005_invariants.py``。
    """
    from dataclasses import replace

    from ai_detector.service.config import ServiceConfigError

    config = _config_for(tmp_path, bundle_path)
    with pytest.raises(ServiceConfigError, match="score_semantics"):
        replace(config, score_semantics="fraud_probability")


def test_runtime_does_not_touch_real_data_paths() -> None:
    """运行时模块不得引用 IEEE-CIS 原始数据或训练入口。"""
    from pathlib import Path as _Path

    source = _Path("src/ai_detector/service/runtime.py").read_text(encoding="utf-8")
    for banned in ("train_transaction", "train_identity", "split_manifest", "isFraud", "fit("):
        assert banned not in source, banned


def test_service_dependency_error_is_a_pydantic_free_error() -> None:
    """``BundleLoadError`` 必须能用普通属性表达原因，便于启动诊断。"""
    error = BundleLoadError("code_x", "detail_y")
    assert error.code == "code_x"
    assert error.detail == "detail_y"
    assert not isinstance(error, ValidationError)
