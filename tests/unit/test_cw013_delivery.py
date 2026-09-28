"""CW-013 离线 bundle 交付的区分性测试：导出、导入、失败关闭与报告安全。

**不触碰真实 bundle**：每个用例都在项目内被忽略的暂存目录
（``artifacts/tmp_tests/cw013/``）里造一份**小型合成** bundle 与配套的门禁配置，
因此导出/导入的信任锚、目标路径、交付包目录全在暂存区内，真实产物逐字节不变。

重点守住的一条：**包内清单不能绕过本地门禁配置的冻结摘要**。用例会构造一个
"载荷被篡改、同时清单也被改成与新字节一致"的包——它自洽，但与配置不符，必须失败。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

import yaml

from ai_detector.config import PROJECT_ROOT
from project_gate import bundle_delivery as bd

STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests" / "cw013"

#: 交付包目录只允许落在被 Git 忽略的这个子树里（`safe_package_dir` 强制）。
DELIVERY_ROOT_REL = "artifacts/delivery"

#: 合成 bundle 的内容（不是真模型，只用于字节级校验路径）。
PAYLOAD_BYTES = b"cw013-synthetic-bundle-payload-" + b"x" * 64


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _complete_manifest(*, payload: bytes, name: str = bd.PAYLOAD_NAME) -> dict:
    """与给定载荷**自洽**的清单（用于构造"完整但不符合配置"的包）。"""
    return {
        "format_version": bd.PACKAGE_FORMAT_VERSION,
        "created_at_utc": "2026-09-27T00:00:00+00:00",
        "payload": {"name": name, "size": len(payload), "sha256": _digest(payload)},
    }


class Stage:
    """暂存区：合成 bundle、门禁配置与交付包目录。

    包目录放在被忽略的 ``artifacts/delivery/`` 下（交付模块强制如此），
    bundle 与配置仍放在 ``artifacts/tmp_tests/`` 下。
    """

    def __init__(self, directory: Path, package_root: Path) -> None:
        self.directory = directory
        self.package_root = package_root
        self.relative = directory.relative_to(PROJECT_ROOT).as_posix()
        self.bundle_rel = f"{self.relative}/model/bundle.joblib"
        self.bundle_path = PROJECT_ROOT / self.bundle_rel
        self.config_rel = f"{self.relative}/quality_gate.yaml"
        self.package_rel = f"{package_root.relative_to(PROJECT_ROOT).as_posix()}/package"
        self.package_path = PROJECT_ROOT / self.package_rel

    def write_bundle(self, data: bytes = PAYLOAD_BYTES) -> str:
        self.bundle_path.parent.mkdir(parents=True, exist_ok=True)
        self.bundle_path.write_bytes(data)
        return _digest(data)

    def write_config(self, *, sha256: str | None = None, path: str | None = None) -> None:
        raw = {
            "quality_gate": {
                "inputs": {
                    "bundle": {
                        "path": path if path is not None else self.bundle_rel,
                        "sha256": sha256 if sha256 is not None else _digest(PAYLOAD_BYTES),
                    }
                },
                "report": {"path": f"{self.relative}/report.json"},
            }
        }
        (PROJECT_ROOT / self.config_rel).write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def export(self, *, package_dir: str | None = None, **kwargs) -> dict:
        return bd.export_bundle(
            root=PROJECT_ROOT,
            package_dir=package_dir if package_dir is not None else self.package_rel,
            config_rel=self.config_rel,
            **kwargs,
        )

    def import_(self, *, package_dir: str | None = None, **kwargs) -> dict:
        return bd.import_bundle(
            root=PROJECT_ROOT,
            package_dir=package_dir if package_dir is not None else self.package_rel,
            config_rel=self.config_rel,
            **kwargs,
        )

    def payload(self) -> Path:
        return self.package_path / bd.PAYLOAD_NAME

    def manifest(self) -> dict:
        return json.loads((self.package_path / bd.MANIFEST_NAME).read_text(encoding="utf-8"))

    def write_manifest(self, manifest: dict) -> None:
        (self.package_path / bd.MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@pytest.fixture
def stage() -> Stage:
    directory = STAGE_ROOT / uuid.uuid4().hex
    (directory / "model").mkdir(parents=True)
    package_root = PROJECT_ROOT / DELIVERY_ROOT_REL / f"cw013_test_{uuid.uuid4().hex}"
    package_root.mkdir(parents=True)
    staged = Stage(directory, package_root)
    staged.write_bundle()
    staged.write_config()
    try:
        yield staged
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        shutil.rmtree(package_root, ignore_errors=True)


# --- 导出 -----------------------------------------------------------------


def test_export_writes_payload_and_manifest(stage: Stage) -> None:
    result = stage.export()
    assert result["status"] == "exported"
    assert result["format_version"] == bd.PACKAGE_FORMAT_VERSION
    assert stage.payload().read_bytes() == PAYLOAD_BYTES
    manifest = stage.manifest()
    assert manifest["format_version"] == bd.PACKAGE_FORMAT_VERSION
    assert manifest["payload"] == {
        "name": bd.PAYLOAD_NAME,
        "size": len(PAYLOAD_BYTES),
        "sha256": _digest(PAYLOAD_BYTES),
    }
    assert manifest["trust_anchor"]["path"] == stage.config_rel
    # 报告只含项目相对路径与摘要，不含绝对路径
    rendered = json.dumps(result, ensure_ascii=False)
    assert str(PROJECT_ROOT) not in rendered


def test_export_is_idempotent_when_the_package_already_matches(stage: Stage) -> None:
    stage.export()
    again = stage.export()
    assert again["status"] == "already_present"


def test_export_refuses_when_the_source_digest_does_not_match(stage: Stage) -> None:
    """源字节与门禁配置冻结摘要不符：不导出，且不留交付包。"""
    stage.write_config(sha256="0" * 64)
    with pytest.raises(bd.DeliveryError) as caught:
        stage.export()
    assert caught.value.code == "source_digest_mismatch"
    assert caught.value.exit_code == bd.EXIT_TRUST
    assert not stage.package_path.exists(), "校验失败不得产出交付包"


def test_export_fails_when_the_source_is_missing(stage: Stage) -> None:
    stage.bundle_path.unlink()
    with pytest.raises(bd.DeliveryError) as caught:
        stage.export()
    assert caught.value.code == "source_missing"
    assert caught.value.exit_code == bd.EXIT_TRUST


def test_export_rejects_a_tampered_payload_inside_an_existing_package(stage: Stage) -> None:
    """已存在包里的载荷被改坏：包内部不再自洽 → 失败关闭，且不改动它。"""
    stage.export()
    stage.payload().write_bytes(b"different bytes")
    with pytest.raises(bd.DeliveryError) as caught:
        stage.export()
    assert caught.value.code == "payload_mismatch"
    assert caught.value.exit_code == bd.EXIT_PACKAGE
    assert stage.payload().read_bytes() == b"different bytes", "拒绝时不得覆盖"


# --- 已有内容必须先是**完整合法**的包（首次验收阻断项 1） ---------------


def test_export_rejects_a_payload_only_package(stage: Stage) -> None:
    """**验收反例**：目录里只有与冻结摘要相符的载荷，没有清单。

    导出**不得**报 `already_present`——那会让"导出成功"产生一个无法导入的包。
    """
    stage.package_path.mkdir(parents=True)
    (stage.package_path / bd.PAYLOAD_NAME).write_bytes(PAYLOAD_BYTES)
    with pytest.raises(bd.DeliveryError) as caught:
        stage.export()
    assert caught.value.code == "package_incomplete"
    assert caught.value.exit_code == bd.EXIT_PACKAGE
    assert not (stage.package_path / bd.MANIFEST_NAME).exists(), "失败不得悄悄补出清单"


def test_export_rejects_a_package_with_extra_entries(stage: Stage) -> None:
    stage.export()
    (stage.package_path / "extra.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(bd.DeliveryError) as caught:
        stage.export()
    assert caught.value.code == "package_extra_entry"


def test_export_rejects_a_corrupt_manifest_in_an_existing_package(stage: Stage) -> None:
    stage.export()
    (stage.package_path / bd.MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(bd.DeliveryError) as caught:
        stage.export()
    assert caught.value.code == "manifest_unreadable"


def test_export_rejects_a_complete_package_that_contradicts_the_config(stage: Stage) -> None:
    """完整但字节与冻结摘要不符的包：拒绝，且**不改动**它的字节。"""
    other = b"a different model" + b"w" * 32
    stage.package_path.mkdir(parents=True)
    (stage.package_path / bd.PAYLOAD_NAME).write_bytes(other)
    stage.write_manifest(_complete_manifest(payload=other))
    before = stage.payload().read_bytes()

    with pytest.raises(bd.DeliveryError) as caught:
        stage.export()
    assert caught.value.code == "trust_digest_mismatch"
    assert caught.value.exit_code == bd.EXIT_TRUST
    assert stage.payload().read_bytes() == before, "拒绝时不得改动既有包字节"


def test_a_successful_export_always_produces_an_importable_package(stage: Stage) -> None:
    """`exported` 与 `already_present` 都必须意味着包**确实可导入**。"""
    assert stage.export()["status"] == "exported"
    # 再导出一次：现在是完整合法的包，因此 already_present —— 并且它必须仍可导入
    assert stage.export()["status"] == "already_present"
    stage.bundle_path.unlink()
    assert stage.import_()["status"] == "imported"
    assert stage.bundle_path.read_bytes() == PAYLOAD_BYTES


# --- 交付包目录必须限定在被忽略的交付区（首次验收阻断项 2） -------------


@pytest.mark.parametrize(
    "candidate",
    [
        "configs/cw013_not_ignored",
        "src/cw013_not_ignored",
        "artifacts/tmp_tests/cw013_not_ignored",
        "artifacts",
        "artifacts/models",
        ".",
    ],
)
def test_package_dir_is_restricted_to_the_ignored_delivery_area(candidate: str) -> None:
    target = PROJECT_ROOT / candidate
    existed = target.exists()
    with pytest.raises(bd.DeliveryError) as caught:
        bd.safe_package_dir(candidate, root=PROJECT_ROOT)
    assert caught.value.exit_code == bd.EXIT_PATH
    assert target.exists() == existed, "拒绝时不得创建或删除任何目录"


def test_delivery_root_itself_is_accepted() -> None:
    resolved = bd.safe_package_dir(DELIVERY_ROOT_REL, root=PROJECT_ROOT)
    assert resolved == (PROJECT_ROOT / DELIVERY_ROOT_REL).resolve()


def test_export_never_writes_into_a_committed_directory(stage: Stage) -> None:
    """不合规的包目录必须在**任何写入之前**被拒绝（只读校验，不往已提交目录写东西）。

    目录名带随机后缀并**自带清理**：即便将来守卫被改坏、真的写了进去，
    也不会在已提交目录里留下残渣（这正是一次变异验证踩到的）。
    """
    for parent in ("configs", "src"):
        candidate = f"{parent}/cw013_reject_{uuid.uuid4().hex}"
        target = PROJECT_ROOT / candidate
        try:
            assert not target.exists()
            with pytest.raises(bd.DeliveryError) as caught:
                stage.export(package_dir=candidate)
            assert caught.value.exit_code == bd.EXIT_PATH
            assert not target.exists(), candidate
        finally:
            shutil.rmtree(target, ignore_errors=True)


def test_import_also_restricts_the_package_directory(stage: Stage) -> None:
    """导入同样只从被忽略的交付区读取，避免"把包放进已提交目录"成为可用路径。"""
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_(package_dir="configs/cw013_not_ignored")
    assert caught.value.exit_code == bd.EXIT_PATH


def test_package_dir_must_stay_project_relative(stage: Stage) -> None:
    with pytest.raises(bd.DeliveryError) as caught:
        stage.export(package_dir=str(PROJECT_ROOT / DELIVERY_ROOT_REL / "abs"))
    assert caught.value.exit_code == bd.EXIT_PATH


# --- 导入：成功与幂等 ----------------------------------------------------


def test_import_creates_the_target_and_is_idempotent(stage: Stage) -> None:
    stage.export()
    stage.bundle_path.unlink()  # 模拟干净检出：目标不存在
    result = stage.import_()
    assert result["status"] == "imported"
    assert stage.bundle_path.read_bytes() == PAYLOAD_BYTES
    assert result["target"]["path"] == stage.bundle_rel
    again = stage.import_()
    assert again["status"] == "already_present"


def test_import_does_not_bypass_the_config_digest_via_the_manifest(stage: Stage) -> None:
    """**核心反例**：载荷被篡改且清单被改成与新字节自洽 —— 仍必须失败。

    清单只是自描述；信任锚是本地门禁配置里的冻结摘要，篡改方改不动它。
    """
    stage.export()
    tampered = b"tampered-payload" + b"y" * 32
    stage.payload().write_bytes(tampered)
    manifest = stage.manifest()
    manifest["payload"] = {
        "name": bd.PAYLOAD_NAME,
        "size": len(tampered),
        "sha256": _digest(tampered),
    }
    stage.write_manifest(manifest)  # 清单与新字节完全自洽

    stage.bundle_path.unlink()
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.code == "trust_digest_mismatch"
    assert caught.value.exit_code == bd.EXIT_TRUST
    assert not stage.bundle_path.exists(), "信任校验失败不得写出目标"


def test_import_rejects_a_payload_that_contradicts_its_manifest(stage: Stage) -> None:
    stage.export()
    stage.payload().write_bytes(b"short")
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.code == "payload_mismatch"
    assert caught.value.exit_code == bd.EXIT_PACKAGE


def test_import_refuses_to_overwrite_a_target_with_different_bytes(stage: Stage) -> None:
    stage.export()
    stage.bundle_path.write_bytes(b"an older, different model")
    before = stage.bundle_path.read_bytes()
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.code == "target_conflict"
    assert caught.value.exit_code == bd.EXIT_TARGET
    assert stage.bundle_path.read_bytes() == before, "拒绝覆盖时原文件必须字节不变"


# --- 导入：坏包结构 ------------------------------------------------------


def test_import_rejects_a_missing_package(stage: Stage) -> None:
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.code == "package_missing"
    assert caught.value.exit_code == bd.EXIT_PACKAGE


def test_import_rejects_an_extra_entry(stage: Stage) -> None:
    stage.export()
    (stage.package_path / "extra.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.code == "package_extra_entry"
    assert caught.value.exit_code == bd.EXIT_PACKAGE


def test_import_rejects_an_incomplete_package(stage: Stage) -> None:
    stage.export()
    (stage.package_path / bd.MANIFEST_NAME).unlink()
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.code == "package_incomplete"


@pytest.mark.parametrize("mutate,expected", [
    ("not json", "manifest_unreadable"),
    (json.dumps({"format_version": "other"}), "manifest_version"),
    (json.dumps({"format_version": bd.PACKAGE_FORMAT_VERSION}), "manifest_payload"),
])
def test_import_rejects_a_bad_manifest(stage: Stage, mutate: str, expected: str) -> None:
    stage.export()
    (stage.package_path / bd.MANIFEST_NAME).write_text(mutate, encoding="utf-8")
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.code == expected


def test_import_rejects_a_manifest_with_a_wrong_payload_name(stage: Stage) -> None:
    stage.export()
    manifest = stage.manifest()
    manifest["payload"]["name"] = "some_other.joblib"
    stage.write_manifest(manifest)
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.code == "manifest_name"


# --- 路径安全 ------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "/absolute/path.joblib",
        "C:/windows/path.joblib",
        "back\\slash.joblib",
        "a/../escape.joblib",
        "a//b.joblib",
        "",
        ".",
        "..",
    ],
)
def test_path_safety_rejects_unsafe_forms(relative: str) -> None:
    with pytest.raises(bd.DeliveryError) as caught:
        bd.safe_project_path(relative, root=PROJECT_ROOT)
    assert caught.value.exit_code == bd.EXIT_PATH


def test_path_safety_accepts_a_normal_project_relative_path() -> None:
    resolved = bd.safe_project_path("configs/service.yaml", root=PROJECT_ROOT)
    assert resolved == (PROJECT_ROOT / "configs" / "service.yaml").resolve()


def test_import_rejects_an_unsafe_target_in_the_config(stage: Stage) -> None:
    """配置里的目标路径越界时必须拒绝导入，且不写出任何东西。"""
    stage.export()
    stage.write_config(path="../outside.joblib")
    with pytest.raises(bd.DeliveryError) as caught:
        stage.import_()
    assert caught.value.exit_code == bd.EXIT_PATH


# --- 配置即信任锚 --------------------------------------------------------


def test_frozen_target_is_read_from_the_gate_config(stage: Stage) -> None:
    target = bd.frozen_target(PROJECT_ROOT, stage.config_rel)
    assert target.relative_path == stage.bundle_rel
    assert target.sha256 == _digest(PAYLOAD_BYTES)


def test_unreadable_config_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(bd.DeliveryError) as caught:
        bd.frozen_target(tmp_path, "configs/quality_gate.yaml")
    assert caught.value.exit_code == bd.EXIT_TRUST


def test_stream_digest_matches_bytes_on_disk(stage: Stage) -> None:
    digest, size = bd.stream_digest(PROJECT_ROOT / stage.bundle_rel)
    assert (digest, size) == (_digest(PAYLOAD_BYTES), len(PAYLOAD_BYTES))
