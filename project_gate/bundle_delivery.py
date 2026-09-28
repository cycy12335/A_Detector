"""CW-013 离线 bundle 交付：导出、导入与按**冻结摘要**校验。

## 信任锚只有一个

导出的**来源**与导入的**验收标准**都取自 CW-009 门禁配置
（``configs/quality_gate.yaml`` → ``quality_gate.inputs.bundle.path`` 与 ``.sha256``）。
交付包里的 ``manifest.json`` 只是自描述，**不能**充当授权来源：

- 导出前先把真实 bundle 的流式 SHA-256 与配置里的冻结值比对，不符就**不导出**；
- 导入时把交付包里的实际字节摘要与**本地配置**的冻结值比对，
  清单里写什么摘要都不改变这个判定。

因此"换一个来源不明的模型"既导不出、也导不进。

## 只写被 Git 忽略的目录

交付包默认落在 ``artifacts/delivery/``（``.gitignore`` 的 ``artifacts/**`` 规则已排除），
模型二进制不进入版本控制。

## 写入是原子的，已有目标不被覆盖

导入先写同目录临时文件再 ``os.replace``，失败清理临时文件。目标已存在时：
字节摘要相同 → 幂等成功（不重写）；不同 → **拒绝**并保持原文件字节不变。

## 错误分类（稳定退出码）

===  ==================================================
码    含义
===  ==================================================
0    成功（新建或已存在且摘要相同）
2    信任摘要不符（导出源不符 / 导入包与配置冻结摘要不符）
3    交付包结构或清单不合格（缺文件、多余路径、清单损坏、大小不符）
4    路径不安全（绝对路径、``..``、反斜杠、盘符、符号链接越界）
5    目标已存在且字节不同，拒绝覆盖
6    写入或报告失败
===  ==================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

#: 项目根：本文件位于 ``<root>/project_gate/bundle_delivery.py``。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 交付包格式版本。清单与结构一起变化时必须升版。
PACKAGE_FORMAT_VERSION = "cw013_bundle_package_v1"

#: 默认交付包目录（项目相对，被 Git 忽略）。
PACKAGE_DIR_REL = "artifacts/delivery/cw013_lightgbm_bundle"

#: 交付包**只允许**落在被 Git 忽略的这个子树之下；``configs/``、``src/`` 等已提交目录一律拒绝。
DELIVERY_ROOT_REL = "artifacts/delivery"

#: 交付包内允许出现的**全部**文件；任何其它条目都判为不合格。
PAYLOAD_NAME = "lightgbm_bundle.joblib"
MANIFEST_NAME = "manifest.json"
ALLOWED_PACKAGE_FILES = frozenset({PAYLOAD_NAME, MANIFEST_NAME})

#: 默认门禁配置（冻结输入的来源）。
QUALITY_GATE_CONFIG_REL = "configs/quality_gate.yaml"

#: 报告目录（项目相对，被 Git 忽略）。
REPORT_DIR_REL = "artifacts/reports/cw013_delivery"

#: 流式校验的分块大小。
CHUNK_SIZE = 1 << 20

#: 稳定退出码。
EXIT_OK = 0
EXIT_TRUST = 2
EXIT_PACKAGE = 3
EXIT_PATH = 4
EXIT_TARGET = 5
EXIT_WRITE = 6


class DeliveryError(RuntimeError):
    """带稳定分类的交付失败。

    ``result`` 携带**已经收集到的安全字段**（相对路径、摘要、大小、状态），
    便于失败时仍然写出可定位的报告；它绝不含 bundle 内容或绝对路径。
    """

    def __init__(self, code: str, detail: str, *, exit_code: int, result: dict | None = None) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.exit_code = int(exit_code)
        self.result = dict(result or {})
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True)
class FrozenTarget:
    """门禁配置冻结的 bundle 目标。"""

    relative_path: str
    sha256: str


def stream_digest(path: Path) -> tuple[str, int]:
    """流式计算 ``(sha256, 字节数)``；不把文件整体读进内存。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def frozen_target(root: Path = PROJECT_ROOT, config_rel: str = QUALITY_GATE_CONFIG_REL) -> FrozenTarget:
    """从门禁配置读取冻结的 bundle 路径与摘要——**唯一**的信任锚。"""
    try:
        raw = yaml.safe_load((root / config_rel).read_text(encoding="utf-8"))
        section = raw["quality_gate"]["inputs"]["bundle"]
        relative = section["path"]
        digest = section["sha256"]
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError) as error:
        raise DeliveryError(
            "config_unreadable",
            f"无法从门禁配置读取冻结 bundle 目标：{type(error).__name__}",
            exit_code=EXIT_TRUST,
        ) from error
    if not isinstance(relative, str) or not isinstance(digest, str) or len(digest) != 64:
        raise DeliveryError(
            "config_invalid", "门禁配置的 bundle 路径或摘要形状非法", exit_code=EXIT_TRUST
        )
    return FrozenTarget(relative_path=relative, sha256=digest)


def safe_project_path(relative: str, *, root: Path = PROJECT_ROOT) -> Path:
    """把项目相对路径解析成绝对路径，并拒绝一切越界写法。

    拒绝：绝对路径/盘符/UNC、反斜杠、空、``.``/``..`` 组件、连续斜杠，
    以及解析后不在 ``root`` 之下或路径上存在符号链接的情况。
    """
    if not isinstance(relative, str) or not relative:
        raise DeliveryError("path_invalid", "路径为空", exit_code=EXIT_PATH)
    if relative.startswith(("/", "\\")) or "\\" in relative or ":" in relative:
        raise DeliveryError("path_invalid", "路径必须是项目相对且不含盘符", exit_code=EXIT_PATH)
    if "//" in relative:
        raise DeliveryError("path_invalid", "路径不得含连续斜杠", exit_code=EXIT_PATH)
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise DeliveryError(
            "path_traversal", "路径不得含空、. 或 .. 组件", exit_code=EXIT_PATH
        )
    resolved_root = root.resolve()
    candidate = resolved_root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise DeliveryError(
                "path_symlink", "路径组件不得是符号链接", exit_code=EXIT_PATH
            )
    resolved = candidate.resolve()
    if resolved != candidate or not str(resolved).startswith(str(resolved_root) + os.sep):
        raise DeliveryError("path_escape", "路径解析后越出项目根", exit_code=EXIT_PATH)
    return resolved


def safe_package_dir(relative: str, *, root: Path = PROJECT_ROOT) -> Path:
    """解析交付包目录，并强制它落在被 Git 忽略的 ``artifacts/delivery/`` 之下。

    交付包里有模型二进制，**绝不能**写进会被提交的目录（``configs/``、``src/`` …）。
    先做通用的项目相对/边界校验，再检查是否位于交付区。

    Raises:
        DeliveryError: 路径不安全（``EXIT_PATH``），或不在交付区内
            （``package_dir_not_ignored``）。
    """
    resolved = safe_project_path(relative, root=root)
    delivery_root = (root / DELIVERY_ROOT_REL).resolve()
    if resolved != delivery_root and delivery_root not in resolved.parents:
        raise DeliveryError(
            "package_dir_not_ignored",
            f"交付包目录必须位于被 Git 忽略的 {DELIVERY_ROOT_REL}/ 之下",
            exit_code=EXIT_PATH,
        )
    return resolved


def _package_dir(root: Path, package_dir: str | None) -> tuple[Path, str]:
    relative = package_dir or PACKAGE_DIR_REL
    return safe_package_dir(relative, root=root), relative


def _relative(root: Path, path: Path) -> str:
    """把绝对路径转成项目相对 POSIX 形式；不在项目内时返回 ``<outside>``。"""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - 调用点已做越界校验
        return "<outside>"


def _write_atomic(path: Path, data: bytes) -> None:
    """同目录临时文件 + ``os.replace``；失败清理临时文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_source(root: Path, target: FrozenTarget) -> tuple[Path, str, int]:
    """校验真实 bundle 存在且字节与冻结摘要一致。"""
    source = safe_project_path(target.relative_path, root=root)
    if not source.is_file():
        raise DeliveryError(
            "source_missing", "真实 bundle 不存在", exit_code=EXIT_TRUST
        )
    digest, size = stream_digest(source)
    if digest != target.sha256:
        raise DeliveryError(
            "source_digest_mismatch",
            "真实 bundle 与门禁配置的冻结摘要不符",
            exit_code=EXIT_TRUST,
        )
    return source, digest, size


def _manifest(
    *, digest: str, size: int, source_relative: str, payload_name: str, config_rel: str, config_sha256: str
) -> dict[str, Any]:
    return {
        "format_version": PACKAGE_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": {"name": payload_name, "size": size, "sha256": digest},
        "source": {"path": source_relative, "size": size, "sha256": digest},
        "trust_anchor": {"path": config_rel, "sha256": config_sha256},
        "note": "清单是自描述，不是授权来源；导入时以本地门禁配置的冻结摘要为准。",
    }


def export_bundle(
    *,
    root: Path = PROJECT_ROOT,
    package_dir: str | None = None,
    config_rel: str = QUALITY_GATE_CONFIG_REL,
) -> dict[str, Any]:
    """把已验收 bundle 导出成离线交付包。

    Raises:
        DeliveryError: 源缺失或与冻结摘要不符（``EXIT_TRUST``）、
            包已存在且字节不同（``EXIT_TARGET``）、写入失败（``EXIT_WRITE``）。
    """
    target = frozen_target(root, config_rel)
    source, digest, size = _check_source(root, target)
    directory, directory_rel = _package_dir(root, package_dir)

    result: dict[str, Any] = {
        "status": "exported",
        "format_version": PACKAGE_FORMAT_VERSION,
        "package_dir": directory_rel,
        "payload": {"name": PAYLOAD_NAME, "size": size, "sha256": digest},
        "source": {"path": _relative(root, source), "size": size, "sha256": digest},
        "trust_anchor": {"path": config_rel, "sha256": digest},
    }

    # 目录里已经有内容时，它必须先是**完整、合法、与冻结配置一致**的包。
    # 只比对载荷摘要是不够的：那样"载荷在、清单不在"会被报成成功，
    # 而它根本导入不了（首次验收的阻断项）。
    if directory.is_dir() and any(directory.iterdir()):
        _validate_package(directory, root=root, target=target)
        result["status"] = "already_present"
        return result

    config_sha256, _ = stream_digest(root / config_rel)
    manifest = _manifest(
        digest=digest,
        size=size,
        source_relative=_relative(root, source),
        payload_name=PAYLOAD_NAME,
        config_rel=config_rel,
        config_sha256=config_sha256,
    )
    payload = directory / PAYLOAD_NAME
    try:
        payload.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(payload, source.read_bytes())
        _write_atomic(
            directory / MANIFEST_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    except OSError as error:
        raise DeliveryError(
            "write_failed", f"写入交付包失败：{type(error).__name__}", exit_code=EXIT_WRITE,
            result=result,
        ) from error
    return result


def _validate_package(directory: Path, *, root: Path, target: FrozenTarget) -> tuple[Path, dict, str, int]:
    """校验交付包结构与清单，并返回 ``(payload, manifest, 实际摘要, 实际大小)``。"""
    if not directory.is_dir():
        raise DeliveryError("package_missing", "交付包目录不存在", exit_code=EXIT_PACKAGE)
    entries = sorted(item.name for item in directory.iterdir())
    unexpected = [name for name in entries if name not in ALLOWED_PACKAGE_FILES]
    if unexpected:
        raise DeliveryError(
            "package_extra_entry",
            f"交付包含 {len(unexpected)} 个不允许的条目",
            exit_code=EXIT_PACKAGE,
        )
    payload = directory / PAYLOAD_NAME
    manifest_path = directory / MANIFEST_NAME
    if not payload.is_file() or not manifest_path.is_file():
        raise DeliveryError(
            "package_incomplete", "交付包缺少载荷或清单", exit_code=EXIT_PACKAGE
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DeliveryError(
            "manifest_unreadable", f"清单不可解析：{type(error).__name__}", exit_code=EXIT_PACKAGE
        ) from error
    if not isinstance(manifest, dict) or manifest.get("format_version") != PACKAGE_FORMAT_VERSION:
        raise DeliveryError(
            "manifest_version", "清单格式版本不符", exit_code=EXIT_PACKAGE
        )
    declared = manifest.get("payload")
    if not isinstance(declared, dict):
        raise DeliveryError("manifest_payload", "清单缺少 payload 段", exit_code=EXIT_PACKAGE)
    if declared.get("name") != PAYLOAD_NAME:
        raise DeliveryError("manifest_name", "清单载荷名不符", exit_code=EXIT_PACKAGE)

    digest, size = stream_digest(payload)
    if declared.get("size") != size or declared.get("sha256") != digest:
        raise DeliveryError(
            "payload_mismatch", "载荷与清单声明的大小或摘要不符", exit_code=EXIT_PACKAGE
        )
    # **信任锚**：本地门禁配置的冻结摘要。清单说什么都不改变这一步。
    if digest != target.sha256:
        raise DeliveryError(
            "trust_digest_mismatch",
            "交付包字节与本地门禁配置的冻结摘要不符",
            exit_code=EXIT_TRUST,
        )
    return payload, manifest, digest, size


def import_bundle(
    *,
    root: Path = PROJECT_ROOT,
    package_dir: str | None = None,
    config_rel: str = QUALITY_GATE_CONFIG_REL,
) -> dict[str, Any]:
    """把离线交付包导入到门禁配置冻结的目标路径。

    Raises:
        DeliveryError: 包结构/清单不合格（``EXIT_PACKAGE``）、
            字节与配置冻结摘要不符（``EXIT_TRUST``）、路径不安全（``EXIT_PATH``）、
            目标已有不同字节（``EXIT_TARGET``）、写入失败（``EXIT_WRITE``）。
    """
    target = frozen_target(root, config_rel)
    directory, directory_rel = _package_dir(root, package_dir)
    payload, _manifest, digest, size = _validate_package(
        directory, root=root, target=target
    )
    destination = safe_project_path(target.relative_path, root=root)

    result: dict[str, Any] = {
        "status": "imported",
        "format_version": PACKAGE_FORMAT_VERSION,
        "package_dir": directory_rel,
        "target": {"path": target.relative_path, "size": size, "sha256": digest},
        "trust_anchor": {"path": config_rel, "sha256": target.sha256},
    }

    if destination.is_file():
        existing, existing_size = stream_digest(destination)
        if existing == digest:
            result["status"] = "already_present"
            return result
        raise DeliveryError(
            "target_conflict",
            "目标已存在且字节不同，拒绝覆盖（原文件保持不变）",
            exit_code=EXIT_TARGET,
            result=result,
        )

    try:
        _write_atomic(destination, payload.read_bytes())
    except OSError as error:
        raise DeliveryError(
            "write_failed", f"写入目标失败：{type(error).__name__}", exit_code=EXIT_WRITE,
            result=result,
        ) from error
    written, written_size = stream_digest(destination)
    if written != target.sha256 or written_size != size:  # pragma: no cover - 防御性
        raise DeliveryError(
            "write_verify_failed", "写入后复核摘要不符", exit_code=EXIT_WRITE, result=result
        )
    return result


def report_path(root: Path, name: str) -> Path:
    """报告路径（项目相对、被 Git 忽略）。"""
    return root / REPORT_DIR_REL / name
