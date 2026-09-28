"""IEEE-CIS 测试夹具构造器。

统一提供"基于仓库内合成 fixture 派生数据变体"的能力，使失败路径测试（缺列、重复键、
非法标签等）不必各自手写 CSV。

所有数据都是**项目自行构造的合成数据**，不访问网络，不包含任何真实交易。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from ai_detector.config import PROJECT_ROOT
from ai_detector.data.ieee_cis import IeeeCisConfig

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ieee_cis"
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"


def read_base_transactions() -> pd.DataFrame:
    """读取仓库内的合成 transaction fixture。"""
    return pd.read_csv(FIXTURE_DIR / TRANSACTION_FILE)


def read_base_identity() -> pd.DataFrame:
    """读取仓库内的合成 identity fixture。"""
    return pd.read_csv(FIXTURE_DIR / IDENTITY_FILE)


class IeeeCisFixtureBuilder:
    """在临时目录中构造 IEEE-CIS 数据变体与配套配置。"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.raw_dir = workspace / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # --- 数据 -----------------------------------------------------------

    def write(
        self,
        *,
        transactions: pd.DataFrame | None = None,
        identity: pd.DataFrame | None = None,
        transaction_name: str = TRANSACTION_FILE,
        identity_name: str = IDENTITY_FILE,
        write_identity: bool = True,
    ) -> Path:
        """写出数据文件，返回 raw 目录。未传入的帧使用仓库合成 fixture。"""
        frame = read_base_transactions() if transactions is None else transactions
        frame.to_csv(self.raw_dir / transaction_name, index=False)
        if write_identity:
            identity_frame = read_base_identity() if identity is None else identity
            identity_frame.to_csv(self.raw_dir / identity_name, index=False)
        return self.raw_dir

    # --- 配置 -----------------------------------------------------------

    def raw_mapping(
        self,
        *,
        transaction_name: str = TRANSACTION_FILE,
        identity_name: str = IDENTITY_FILE,
        output_dir: str = "reports",
        **file_overrides: object,
    ) -> dict:
        """基于仓库配置生成指向临时目录的原始映射。

        配置对象与配置文件都由这一份映射派生，避免"测试里手写 YAML 键名"与真实加载器
        期望的键名不一致（曾经因此漏掉过 dataset.id）。
        """
        with (PROJECT_ROOT / "configs" / "ieee_cis.yaml").open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        raw["files"]["raw_dir"] = str(self.raw_dir)
        raw["files"]["transactions"]["name"] = transaction_name
        raw["files"]["identity"]["name"] = identity_name
        for key, value in file_overrides.items():
            raw["files"][key] = value

        report_dir = self.workspace / output_dir
        raw["output"] = {
            "report_dir": str(report_dir),
            "audit_json": str(report_dir / "audit.json"),
            "audit_markdown": str(report_dir / "audit.md"),
            "column_profile_csv": str(report_dir / "column_profile.csv"),
            "split_manifest": str(report_dir / "split_manifest.json"),
            "processed_dir": str(self.workspace / "processed"),
        }
        return raw

    def config(self, **kwargs: object) -> IeeeCisConfig:
        """基于仓库配置生成指向临时目录的配置对象。"""
        return IeeeCisConfig.from_mapping(self.raw_mapping(**kwargs))

    def write_config_file(self, name: str = "ieee_cis_test.yaml", **kwargs: object) -> Path:
        """把同一份映射写成 YAML 文件，供命令行入口测试使用。"""
        path = self.workspace / name
        path.write_text(
            yaml.safe_dump(self.raw_mapping(**kwargs), allow_unicode=True), encoding="utf-8"
        )
        return path


@pytest.fixture
def ieee_cis_fixture(tmp_path: Path) -> IeeeCisFixtureBuilder:
    """返回在临时目录中构造 IEEE-CIS 数据变体的构造器。"""
    return IeeeCisFixtureBuilder(tmp_path)


@pytest.fixture
def real_data_gate_config() -> IeeeCisConfig:
    """指向真实数据目录的配置，用于验证"文件缺失时快速失败"。"""
    with (PROJECT_ROOT / "configs" / "ieee_cis.yaml").open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["files"] = {**raw["files"], "raw_dir": "data/raw/ieee_cis"}
    return IeeeCisConfig.from_mapping(raw)


@pytest.fixture
def cw003_fixture(tmp_path: Path):
    """CW-003 合成夹具构造器（定义见 ``tests/cw003_fixtures.py``）。

    延迟导入：构造器依赖 CW-002 的审计代码，放在函数内可避免 conftest 的导入顺序问题。
    """
    from tests.cw003_fixtures import Cw003FixtureBuilder

    return Cw003FixtureBuilder(tmp_path)


@pytest.fixture
def cw004_fixture(tmp_path: Path):
    """CW-004 合成夹具构造器（定义见 ``tests/cw004_fixtures.py``）。

    继承 CW-003 夹具，并额外用真实的 CW-003 训练代码产出「已验收产物」，
    使 CW-004 的前置门禁在测试里也走真实路径。
    """
    from tests.cw004_fixtures import Cw004FixtureBuilder

    return Cw004FixtureBuilder(tmp_path)
