"""审计命令行入口集成测试。

以子进程方式运行 README 中给出的命令，确保文档里的命令真的能跑通，而不只是函数能跑通。
使用项目自建的合成 fixture，不访问网络，不读取真实 Kaggle 数据。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_detector.config import PROJECT_ROOT
from tests.conftest import IeeeCisFixtureBuilder

AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_ieee_cis.py"


def run_audit_script(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """在子进程中运行审计脚本。

    固定子进程 stdout/stderr 为 UTF-8：Windows 默认使用控制台代码页（中文系统为 cp936），
    按系统默认解码会让中文错误信息变成乱码，导致断言在正确的代码上失败。
    """
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
        env=environment,
    )


def test_audit_script_succeeds_on_fixture(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    ieee_cis_fixture.write()
    config_path = ieee_cis_fixture.write_config_file()

    completed = run_audit_script("--config", str(config_path), cwd=PROJECT_ROOT)

    assert completed.returncode == 0, completed.stderr
    assert "IEEE-CIS Fraud Detection" in completed.stdout
    assert "SHA-256" in completed.stdout
    assert "不代表真实银行生产效果" in completed.stdout

    report_dir = ieee_cis_fixture.workspace / "reports"
    for name in ("audit.json", "audit.md", "column_profile.csv", "split_manifest.json"):
        assert (report_dir / name).is_file(), name


def test_audit_script_reports_missing_files_with_exit_code_2(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    """缺少原始数据时必须以专用退出码失败，并列出放置步骤。"""
    ieee_cis_fixture.raw_dir.mkdir(parents=True, exist_ok=True)
    config_path = ieee_cis_fixture.write_config_file()

    completed = run_audit_script("--config", str(config_path), cwd=PROJECT_ROOT)

    assert completed.returncode == 2
    assert "缺少必需的 IEEE-CIS 原始数据文件" in completed.stderr
    assert "train_transaction.csv" in completed.stderr
    assert "train_identity.csv" in completed.stderr
    assert "https://www.kaggle.com/competitions/ieee-fraud-detection/data" in completed.stderr
    assert str(PROJECT_ROOT) not in completed.stderr, "错误信息不得含机器专属绝对路径"
    assert not (ieee_cis_fixture.workspace / "reports" / "audit.json").exists()


def test_audit_script_skip_row_partitions_flag(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    ieee_cis_fixture.write()
    config_path = ieee_cis_fixture.write_config_file()

    completed = run_audit_script(
        "--config", str(config_path), "--skip-row-partitions", cwd=PROJECT_ROOT
    )

    assert completed.returncode == 0, completed.stderr
    assert not (ieee_cis_fixture.workspace / "processed" / "split_manifest_rows.csv").exists()


def test_audit_script_is_reproducible(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    """同一输入重复运行，除生成时间与耗时外结论必须一致。"""
    ieee_cis_fixture.write()
    config_path = ieee_cis_fixture.write_config_file()
    report_path = ieee_cis_fixture.workspace / "reports" / "audit.json"

    assert run_audit_script("--config", str(config_path), cwd=PROJECT_ROOT).returncode == 0
    first = json.loads(report_path.read_text(encoding="utf-8"))

    assert run_audit_script("--config", str(config_path), cwd=PROJECT_ROOT).returncode == 0
    second = json.loads(report_path.read_text(encoding="utf-8"))

    for payload in (first, second):
        payload.pop("generated_at", None)
        payload.pop("timing_seconds", None)
    assert first == second


def test_audit_script_rejects_missing_config(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    completed = run_audit_script(
        "--config", str(ieee_cis_fixture.workspace / "nope.yaml"), cwd=PROJECT_ROOT
    )

    assert completed.returncode != 0
    assert "配置文件不存在" in (completed.stderr + completed.stdout)


def test_audit_script_runs_from_other_working_directory(
    ieee_cis_fixture: IeeeCisFixtureBuilder, tmp_path: Path
) -> None:
    """从其他目录调用时，用绝对配置路径必须仍然可用（README 说明的行为）。"""
    ieee_cis_fixture.write()
    config_path = ieee_cis_fixture.write_config_file()
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()

    completed = run_audit_script("--config", str(config_path), cwd=other_cwd)

    assert completed.returncode == 0, completed.stderr
    assert (ieee_cis_fixture.workspace / "reports" / "audit.json").is_file()


def test_audit_script_never_downloads_or_reads_credentials(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    """脚本不得依赖网络：源码中不应出现网络客户端或凭据读取。"""
    source = AUDIT_SCRIPT.read_text(encoding="utf-8")
    package_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "src" / "ai_detector" / "data").glob("*.py")
    )

    for forbidden in ("requests", "urllib", "http.client", "socket", "kaggle", "KAGGLE_KEY",
                      "os.environ.get(\"KAGGLE"):
        assert forbidden not in source, f"审计脚本不应引用 {forbidden}"
        assert forbidden not in package_sources, f"数据层不应引用 {forbidden}"


@pytest.mark.parametrize("flag", ["--help"])
def test_audit_script_help(flag: str) -> None:
    completed = run_audit_script(flag, cwd=PROJECT_ROOT)

    assert completed.returncode == 0
    assert "--config" in completed.stdout
