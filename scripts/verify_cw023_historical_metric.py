"""只读核对 CW-004 历史 F1 报告与历史配置指纹；不训练、不读取原始数据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "reports" / "cw004_lightgbm" / "report.json"
CURRENT_CONFIG = ROOT / "configs" / "ieee_cis_lightgbm.yaml"
HISTORICAL_CONFIG_SHA = "901517448045ee00efa0684d3e7a1bd86d975e9ade31828bc67cf539d75d5582"
EXPECTED_CURRENT_CONFIG_SHA = "17d629c8d27b40031385a61f924ca4d842bade1540feb46fec9c161eb144a203"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    historical_config_sha = report["config"]["config_sha256"]
    evaluation = report["test_evaluation"]
    metrics = evaluation["metrics"]
    current_config_sha = sha256(CURRENT_CONFIG)
    checks = {
        "historical_config_sha_matches_report": historical_config_sha == HISTORICAL_CONFIG_SHA,
        "current_config_sha_matches_workspace": current_config_sha == EXPECTED_CURRENT_CONFIG_SHA,
        "partition_is_test": evaluation["partition"] == "test",
        "sample_count_is_118108": metrics["n_samples"] == 118108,
        "f1_is_0.38354854136776667": metrics["f1"] == 0.38354854136776667,
    }
    if not all(checks.values()):
        print(json.dumps({"status": "failed", "checks": checks}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": "read_only_report_verification",
                "historical_config_sha": historical_config_sha,
                "current_config_sha": current_config_sha,
                "partition": evaluation["partition"],
                "n_samples": metrics["n_samples"],
                "f1": metrics["f1"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
