"""CW-004 合成夹具：在临时目录中构造 CW-002 产物、CW-003 已验收产物与 CW-004 配置。

设计要点：

- **不重写 CW-003 夹具**：本构造器继承 :class:`Cw003FixtureBuilder`，复用它的合成数据、
  CW-002 审计与 CW-003 配置构造，只在数据上注入一点**与标签相关**的信号，使 LightGBM
  在合成数据上真的会分裂（否则重要性全为 0，聚合守恒测试会退化成同义反复）。
- **CW-003 产物由真实代码产出**：CW-004 的前置门禁要求 CW-003 的 `report.json`、
  `feature_manifest.json` 与三类模型产物存在且指纹匹配。夹具直接调用 CW-003 的
  ``run_training`` 生成它们，而不是手写 JSON——避免夹具与真实产物格式逐渐脱节。
- 全部数据为项目自建，不包含任何真实 Kaggle 记录，不访问网络，不依赖大文件。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tests.cw003_fixtures import (
    Cw003FixtureBuilder,
    build_synthetic_identity,
    build_synthetic_transactions,
)

#: 合成数据上注入信号所用的种子（与配置种子独立，固定即可）。
SIGNAL_SEED = 20240924


class Cw004FixtureBuilder(Cw003FixtureBuilder):
    """在临时目录中构造一整套 CW-004 可运行环境。"""

    def __init__(
        self,
        workspace: Path,
        *,
        identity_coverage: float = 0.6,
        max_cardinality: int = 50,
        with_signal: bool = True,
    ) -> None:
        super().__init__(
            workspace, identity_coverage=identity_coverage, max_cardinality=max_cardinality
        )
        self.with_signal = with_signal

    # --- 数据：在保留列集合不变的前提下注入与标签相关的信号 ---------------

    def write_data(self) -> Path:
        transactions = build_synthetic_transactions()
        if self.with_signal:
            fraud = transactions["isFraud"].to_numpy()
            rng = np.random.default_rng(SIGNAL_SEED)
            # NumG 与 AmtA 都已在 CW-003 允许清单内，因此列集合与列决策规则完全不变；
            # 只是让它们的分布依赖标签，模型才有可分裂的结构。
            transactions["NumG"] = np.where(
                fraud == 1,
                rng.normal(2.5, 1.0, len(fraud)),
                rng.normal(0.0, 1.0, len(fraud)),
            )
            transactions["AmtA"] = transactions["AmtA"].to_numpy() * np.where(
                fraud == 1, 4.0, 1.0
            )
        identity = build_synthetic_identity(
            transactions["TransactionID"].to_numpy(), coverage=self.identity_coverage
        )
        transactions.to_csv(self.raw_dir / "train_transaction.csv", index=False)
        identity.to_csv(self.raw_dir / "train_identity.csv", index=False)
        return self.raw_dir

    # --- CW-003 已验收产物 + CW-004 配置 ---------------------------------

    def build_cw004(
        self,
        *,
        chunk_size: int = 64,
        prerequisite_status: str = "passed",
    ) -> tuple[Path, object, dict]:
        """构造并返回 ``(CW-004 配置路径, LightGbmConfig, CW-003 训练结果)``。"""
        from ai_detector.ieee_training import run_training
        from ai_detector.lightgbm_training import LightGbmConfig

        cw003_config_path, cw003_config = self.build()
        cw003_result = run_training(cw003_config, chunk_size=chunk_size)

        cw003_report_dir = self.workspace / "out"
        cw003_model_dir = self.workspace / "models"

        # CW-003 协议任务状态：CW-004 门禁要求为 passed。
        task_dir = self.workspace / "handoff" / "TASK-CW003"
        task_dir.mkdir(parents=True, exist_ok=True)
        status_path = task_dir / "status.json"
        status_path.write_text(
            json.dumps(
                {
                    "task_id": "TASK-FIXTURE-CW003",
                    "status": prerequisite_status,
                    "owner": "fixture",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        outputs = {
            "report_dir": str(self.workspace / "cw004_out"),
            "report_json": str(self.workspace / "cw004_out" / "report.json"),
            "report_markdown": str(self.workspace / "cw004_out" / "report.md"),
            "threshold_curve_csv": str(self.workspace / "cw004_out" / "threshold_curve.csv"),
            "comparison_csv": str(self.workspace / "cw004_out" / "comparison.csv"),
            "feature_importance_csv": str(
                self.workspace / "cw004_out" / "feature_importance.csv"
            ),
            "model_dir": str(self.workspace / "cw004_models"),
            "bundle": str(self.workspace / "cw004_models" / "lightgbm_bundle.joblib"),
        }

        manifest_path = cw003_report_dir / "feature_manifest.json"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

        raw = {
            "work_package": "CW-004",
            "seed": 7,
            "cw003": {
                "work_package": "CW-003",
                "config_path": str(cw003_config_path),
                "expected_config_sha256": cw003_config.config_sha256,
                "feature_manifest_path": str(manifest_path),
                "expected_feature_manifest_sha256": _sha256(manifest_path),
                "expected_kept_numeric": len(manifest["kept"]["numeric"]),
                "expected_kept_categorical": len(manifest["kept"]["categorical"]),
                "expected_transformed_dimension": int(manifest["transformed_dimension"]),
                "expected_output_format": str(manifest["output_format"]),
                "report_path": str(cw003_report_dir / "report.json"),
                "baseline_keys": ["rule_tail_anomaly", "dummy", "logistic_regression"],
                "required_artifacts": [
                    str(cw003_model_dir / "logistic_regression_pipeline.joblib"),
                    str(cw003_model_dir / "dummy_pipeline.joblib"),
                    str(cw003_model_dir / "rule_baseline_state.json"),
                ],
                "prerequisite_task_status": str(status_path),
                "prerequisite_required_status": "passed",
            },
            "model": {
                "type": "LGBMClassifier",
                "params": {
                    "boosting_type": "gbdt",
                    "objective": "binary",
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "max_depth": -1,
                    "min_child_samples": 20,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.0,
                    "random_state": 7,
                    "n_jobs": 1,
                    "deterministic": True,
                    "force_col_wise": True,
                    "verbosity": -1,
                },
                "imbalance": {
                    "strategy": "scale_pos_weight",
                    "formula": "train_negative_count / train_positive_count",
                },
                "early_stopping": {
                    "validation_partition": "validation",
                    "metric": "binary_logloss",
                    "stopping_rounds": 50,
                    "first_metric_only": True,
                },
                "api_compatibility_notes": ["夹具：早停走 callback API。"],
                "param_source": "fixture",
            },
            "determinism": {
                "absolute_tolerance": 1.0e-12,
                "relative_tolerance": 0.0,
                "compared_items": ["best_iteration", "validation_scores"],
            },
            "importance": {
                "types": ["gain", "split"],
                "aggregation": "source_feature",
                "top_n_in_report": 30,
                "forbidden_analyses": ["SHAP", "Permutation Importance"],
            },
            "paths": {"outputs": outputs},
            "limits": {"evaluation_partition": "test", "single_shot_test": True},
        }

        config_path = self.workspace / "ieee_cis_lightgbm_test.yaml"
        config_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )
        config = LightGbmConfig.from_yaml(config_path, allow_external_paths=True)
        return config_path, config, {"result": cw003_result, "config": cw003_config}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signal_frame(rows: int = 200, *, seed: int = 5) -> pd.DataFrame:
    """构造一个带少量数值/类别列与二值标签的小帧，供 bundle 往返与打分测试使用。"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "NumA": rng.normal(size=rows),
            "NumB": rng.normal(size=rows),
            "CatA": rng.choice(["a", "b", "c"], rows),
        }
    )
