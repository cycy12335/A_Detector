"""pytest 根配置。

把 ``src`` 加入 ``sys.path``，使 ``python -m pytest`` 在未执行
``pip install -e .`` 的干净环境中也能导入 ``ai_detector``。
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path
from typing import Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

#: CW-006 测试暂存目录。刻意放在**项目内**：生产的政策加载器只接受项目相对路径
#: （硬安全契约，不为测试开后门），因此测试也走同一条真实链路；该路径已被
#: ``.gitignore`` 的 ``artifacts/**`` 规则排除，不会进入提交清单。
STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests"

#: 默认决策数据库（运行时产物，被 ``.gitignore`` 排除）。
DEFAULT_DB_REL = "artifacts/runtime/cw007_decisions.sqlite3"


def _digest_or_none(path: Path) -> tuple[str, int] | None:
    """文件的 ``(sha256, 字节数)``；不存在时为 ``None``。"""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


@pytest.fixture(scope="session", autouse=True)
def guard_default_database() -> Iterator[None]:
    """整轮测试不得**改写**、也不得凭空**创建**默认决策数据库。

    CW-014 把这条不变量的表述从"默认库必须等于某个固定字节摘要"改成会话级守卫。
    原因：默认库是运行时产物、又被 ``.gitignore`` 排除，干净检出里本来就不存在；
    断言固定摘要等于要求 CI 提供**开发机上的那份数据库**，而那既不可复现、
    也与"默认数据库不进入 Git"冲突。

    守卫在两种环境下都成立，而且比原来的断言更强：原来只检查"存在时摘要对不对"，
    现在同时检查"不存在时整轮结束后仍然不存在"——即整套测试都不会在干净检出里
    意外造出一个默认库。
    """
    path = PROJECT_ROOT / DEFAULT_DB_REL
    before = _digest_or_none(path)
    yield
    after = _digest_or_none(path)
    assert after == before, (
        f"整轮测试改写了默认决策数据库（{DEFAULT_DB_REL}）："
        f"before={before} after={after}"
    )


@pytest.fixture(scope="session", autouse=True)
def _clean_test_stage_root():
    """会话结束后清理项目内的测试暂存目录，不在仓库里留下临时文件。"""
    yield
    shutil.rmtree(STAGE_ROOT, ignore_errors=True)

#: CW-007 测试数据库所在目录（项目内，已被 .gitignore 的 artifacts/** 排除）。
STORAGE_TEST_ROOT = PROJECT_ROOT / "artifacts" / "runtime"


def _purge_test_databases() -> None:
    """删除本套件产生的测试数据库及其 sidecar。

    只删 ``cw007_test_`` 前缀的文件——**绝不**触碰生产默认库 ``cw007_decisions.sqlite3``，
    也不碰该目录下任何别的东西。
    """
    if not STORAGE_TEST_ROOT.is_dir():
        return
    for path in STORAGE_TEST_ROOT.glob("cw007_test_*"):
        try:
            if path.is_file():
                path.unlink()
        except OSError:  # pragma: no cover - 仍被占用时留给下次运行清理
            pass


@pytest.fixture(scope="session", autouse=True)
def _clean_test_databases():
    """会话开始与结束各清理一次测试数据库，避免跨运行互相污染。"""
    _purge_test_databases()
    yield
    _purge_test_databases()
