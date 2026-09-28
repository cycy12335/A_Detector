"""文件指纹测试：流式 SHA-256、字节数、相对路径保护。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ai_detector.data.fingerprint import (
    FingerprintError,
    fingerprint_file,
    relative_posix_path,
    sha256_of_file,
)


def test_sha256_matches_hashlib_on_known_content(tmp_path: Path) -> None:
    """指纹必须等于标准库对同样字节的计算结果。"""
    payload = b"TransactionID,TransactionDT,isFraud\n1,100,0\n"
    path = tmp_path / "sample.csv"
    path.write_bytes(payload)

    assert sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"x" * 5000)

    first = sha256_of_file(path)
    second = sha256_of_file(path)

    assert first == second
    assert len(first) == 64


def test_streaming_matches_single_pass_across_chunk_sizes(tmp_path: Path) -> None:
    """不同读取块大小必须得到同一摘要，否则流式实现有边界错误。"""
    path = tmp_path / "sample.bin"
    path.write_bytes(bytes(range(256)) * 400)  # 102400 字节

    reference = sha256_of_file(path, chunk_size=102400)
    for chunk_size in (1, 7, 1024, 4096, 65536):
        assert sha256_of_file(path, chunk_size=chunk_size) == reference, chunk_size


def test_fingerprint_records_size_and_relative_path(tmp_path: Path) -> None:
    payload = b"a,b,c\n1,2,3\n"
    path = tmp_path / "raw" / "train_transaction.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    fingerprint = fingerprint_file(path, root=tmp_path)

    assert fingerprint.size_bytes == len(payload)
    assert fingerprint.file_name == "train_transaction.csv"
    assert fingerprint.relative_path == "raw/train_transaction.csv"
    assert fingerprint.sha256 == hashlib.sha256(payload).hexdigest()
    assert not Path(fingerprint.relative_path).is_absolute()


def test_strict_path_helper_rejects_paths_outside_root(tmp_path: Path) -> None:
    """严格版本对根目录外的路径直接报错。"""
    outside = tmp_path.parent / "outside_project.csv"
    outside.write_bytes(b"data")

    with pytest.raises(FingerprintError, match="拒绝记录绝对路径"):
        relative_posix_path(outside, tmp_path)


def test_fingerprint_degrades_to_file_name_outside_root(tmp_path: Path) -> None:
    """根目录外的文件退化为仅文件名，并显式标注，绝不写入绝对路径。"""
    outside = tmp_path.parent / "outside_project.csv"
    outside.write_bytes(b"data")

    fingerprint = fingerprint_file(outside, root=tmp_path)

    assert fingerprint.path_within_root is False
    assert fingerprint.relative_path == "outside_project.csv"
    assert not Path(fingerprint.relative_path).is_absolute()
    assert str(tmp_path) not in fingerprint.relative_path


def test_fingerprint_flags_paths_inside_root(tmp_path: Path) -> None:
    inside = tmp_path / "data" / "train_transaction.csv"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"a,b\n1,2\n")

    fingerprint = fingerprint_file(inside, root=tmp_path)

    assert fingerprint.path_within_root is True
    assert fingerprint.relative_path == "data/train_transaction.csv"


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FingerprintError, match="文件不存在"):
        sha256_of_file(tmp_path / "nope.csv")
    with pytest.raises(FingerprintError, match="文件不存在"):
        fingerprint_file(tmp_path / "nope.csv", root=tmp_path)


def test_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FingerprintError, match="不是普通文件"):
        sha256_of_file(tmp_path)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_raises(tmp_path: Path, chunk_size: int) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")

    with pytest.raises(FingerprintError, match="chunk_size"):
        sha256_of_file(path, chunk_size=chunk_size)


def test_repeated_runs_are_byte_identical(tmp_path: Path) -> None:
    """同一内容重复写入后重新指纹，结果必须完全一致。"""
    payload = b"same content"
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_bytes(payload)
    second.write_bytes(payload)

    assert fingerprint_file(first, root=tmp_path).sha256 == fingerprint_file(
        second, root=tmp_path
    ).sha256
