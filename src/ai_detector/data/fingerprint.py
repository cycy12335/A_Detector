"""文件指纹：流式计算 SHA-256 与字节数。

设计要点：

- **不把整个文件读入内存**。IEEE-CIS 的 transaction 文件接近 700 MB，一次性读入会与
  后续审计争抢内存；这里固定按块读取。
- **绝不记录机器专属绝对路径**。路径优先表示为相对项目根目录的形式；当文件位于项目
  根目录之外（使用者可能把约 700 MB 数据放在别的盘）时退化为**仅文件名**，并在指纹里
  用 ``path_within_root=False`` 显式标注，使退化可见而不是静默发生。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: 读取块大小。512 KiB 在吞吐与内存之间取折中。
DEFAULT_CHUNK_SIZE = 512 * 1024


class FingerprintError(RuntimeError):
    """无法计算指纹时抛出（文件缺失、路径是目录等）。"""


@dataclass(frozen=True)
class FileFingerprint:
    """一个数据文件的可追溯指纹。"""

    relative_path: str
    file_name: str
    sha256: str
    size_bytes: int
    #: ``relative_path`` 是否真的相对于项目根目录。为 ``False`` 时它只是文件名，
    #: 调用方必须在报告中标注，避免读者误以为它是一条可解析的相对路径。
    path_within_root: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_posix_path(path: str | Path, root: str | Path) -> str:
    """返回相对 ``root`` 的 POSIX 风格路径；路径在 ``root`` 之外时抛出异常。

    适用于调用方**要求**路径必须位于 ``root`` 之内、不接受退化的场景。
    """
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as error:
        raise FingerprintError(
            f"路径不在项目根目录内，拒绝记录绝对路径：{resolved.name}"
        ) from error
    return relative.as_posix()


def report_path(path: str | Path, root: str | Path) -> tuple[str, bool]:
    """返回 ``(用于报告的路径, 是否位于 root 之内)``。

    这是全项目唯一的路径上报策略实现，避免指纹模块与配置模块各自为政：

    - 位于 ``root`` 之内 → 相对 POSIX 路径，标志为 ``True``。
    - 位于 ``root`` 之外 → **仅文件名**，标志为 ``False``。绝不返回绝对路径。
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix(), True
    except ValueError:
        return resolved.name, False


def sha256_of_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """流式计算文件的 SHA-256 十六进制摘要。

    Raises:
        FingerprintError: 文件不存在、不是普通文件，或 ``chunk_size`` 非法。
    """
    if chunk_size < 1:
        raise FingerprintError(f"chunk_size 必须为正整数，实际为 {chunk_size}")

    file_path = Path(path)
    if not file_path.is_file():
        raise FingerprintError(f"无法计算指纹，文件不存在或不是普通文件：{file_path.name}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fingerprint_file(
    path: str | Path,
    *,
    root: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> FileFingerprint:
    """一次流式遍历同时得到 SHA-256 与字节数。

    ``size_bytes`` 由读取到的块累加得出，因此不需要额外 ``stat`` 调用，
    也保证"记录的字节数"与"实际参与哈希的字节数"完全一致。

    路径在 ``root`` 之外时退化为仅文件名，并用 ``path_within_root=False`` 标注。
    """
    if chunk_size < 1:
        raise FingerprintError(f"chunk_size 必须为正整数，实际为 {chunk_size}")

    file_path = Path(path)
    if not file_path.is_file():
        raise FingerprintError(f"无法计算指纹，文件不存在或不是普通文件：{file_path.name}")

    digest = hashlib.sha256()
    total_bytes = 0
    with file_path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
            total_bytes += len(block)

    reported_path, within_root = report_path(file_path, root)
    return FileFingerprint(
        relative_path=reported_path,
        file_name=file_path.name,
        sha256=digest.hexdigest(),
        size_bytes=total_bytes,
        path_within_root=within_root,
    )
