"""命令行入口集成测试。

直接以子进程方式运行 README 中给出的两条命令，确保**文档里的命令真的能跑通**，
而不是只在单元测试里能跑通。数据规模保持很小，不使用网络。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: README 中记录的核心命令（此处以等价的较小行数运行，避免测试变慢）。
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_data.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_baseline.py"


def run_script(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """在子进程中运行脚本。

    显式把子进程的 stdout/stderr 固定为 UTF-8：Windows 默认使用控制台代码页
    （中文系统为 cp936），若按系统默认解码，中文错误信息会变成乱码，导致断言
    在正确的代码上失败。这里只影响测试读取方式，不改变脚本自身行为。
    """
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
        env=environment,
    )


@pytest.fixture(scope="module")
def cli_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cw001_cli")


@pytest.fixture(scope="module")
def generated_csv(cli_workspace: Path) -> Path:
    csv_path = cli_workspace / "data" / "transactions.csv"
    completed = run_script(
        GENERATE_SCRIPT,
        "--seed",
        "42",
        "--rows",
        "600",
        "--output",
        str(csv_path),
        cwd=cli_workspace,
    )
    assert completed.returncode == 0, completed.stderr
    return csv_path


def test_generate_data_script_succeeds(generated_csv: Path) -> None:
    assert generated_csv.is_file()

    header = generated_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "transaction_id,timestamp,user_id,device_id,amount,"
        "transaction_count_1h,is_new_device,is_fraud"
    )
    assert len(generated_csv.read_text(encoding="utf-8").splitlines()) == 601


def test_generate_data_script_is_reproducible(cli_workspace: Path) -> None:
    """同一命令执行两次必须产出逐字节相同的文件。"""
    first = cli_workspace / "repeat_a.csv"
    second = cli_workspace / "repeat_b.csv"

    for target in (first, second):
        completed = run_script(
            GENERATE_SCRIPT,
            "--seed",
            "1234",
            "--rows",
            "200",
            "--output",
            str(target),
            cwd=cli_workspace,
        )
        assert completed.returncode == 0, completed.stderr

    assert first.read_bytes() == second.read_bytes()


def test_generate_data_script_rejects_invalid_rows(cli_workspace: Path) -> None:
    completed = run_script(
        GENERATE_SCRIPT,
        "--rows",
        "0",
        "--output",
        str(cli_workspace / "invalid.csv"),
        cwd=cli_workspace,
    )
    assert completed.returncode != 0
    assert "rows" in (completed.stderr + completed.stdout)


def test_train_script_succeeds_and_writes_artifacts(
    cli_workspace: Path, generated_csv: Path
) -> None:
    config_path = cli_workspace / "baseline_test.yaml"
    report_path = cli_workspace / "artifacts" / "reports" / "report.json"
    model_dir = cli_workspace / "artifacts" / "models"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                f"input_path: {generated_csv.as_posix()}",
                f"report_path: {report_path.as_posix()}",
                f"model_dir: {model_dir.as_posix()}",
                "split:",
                "  train_ratio: 0.6",
                "  validation_ratio: 0.2",
                "threshold_grid:",
                *[f"  - {round(0.05 * step, 2)}" for step in range(1, 20)],
                "models:",
                "  dummy:",
                "    type: dummy_classifier",
                "    params:",
                "      strategy: prior",
                "  logistic_regression:",
                "    type: logistic_regression",
                "    params:",
                "      class_weight: balanced",
                "      max_iter: 1000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    completed = run_script(TRAIN_SCRIPT, "--config", str(config_path), cwd=cli_workspace)
    assert completed.returncode == 0, completed.stderr

    assert report_path.is_file()
    assert (model_dir / "dummy_pipeline.joblib").is_file()
    assert (model_dir / "logistic_regression_pipeline.joblib").is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["work_package"] == "CW-001"
    assert set(report["test_metrics"]) == {"dummy", "logistic_regression"}
    assert report["split"]["sizes"] == {"train": 360, "validation": 120, "test": 120}


def test_train_script_fails_clearly_without_input(cli_workspace: Path) -> None:
    """缺少输入数据时必须给出可读错误，而不是静默产出空报告。"""
    config_path = cli_workspace / "missing_input.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                f"input_path: {(cli_workspace / 'nope.csv').as_posix()}",
                f"report_path: {(cli_workspace / 'report.json').as_posix()}",
                f"model_dir: {(cli_workspace / 'models').as_posix()}",
                "split:",
                "  train_ratio: 0.6",
                "  validation_ratio: 0.2",
                "threshold_grid:",
                "  - 0.5",
                "models:",
                "  dummy:",
                "    type: dummy_classifier",
                "  logistic_regression:",
                "    type: logistic_regression",
                "",
            ]
        ),
        encoding="utf-8",
    )

    completed = run_script(TRAIN_SCRIPT, "--config", str(config_path), cwd=cli_workspace)

    assert completed.returncode != 0
    assert "不存在" in (completed.stderr + completed.stdout)
    assert not (cli_workspace / "report.json").exists()
