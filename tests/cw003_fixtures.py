"""CW-003 合成夹具：构造建模数据、CW-002 审计产物与 CW-003 配置。

设计要点：

- 合成数据的列**刻意覆盖每一条训练期过滤规则**（常量、全缺失、高基数、高缺失率），
  这样"排序/排除"逻辑一旦写错，测试会立刻失败。
- CW-002 的行级切分清单、audit.json 与 split_manifest.json 由 **CW-002 的真实审计代码**
  在合成数据上产出，而不是手写——避免夹具与真实产物的格式逐渐脱节。
- 全部数据为项目自建，不包含任何真实 Kaggle 记录，不访问网络。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ai_detector.config import PROJECT_ROOT

#: 覆盖每条过滤规则的数值列。
CW003_NUMERIC = (
    "AmtA",   # 无缺失；规则基线字段
    "AmtB",   # 无缺失；规则基线字段
    "NumC",   # 10% 缺失 -> 保留，并产生缺失指示器
    "NumD",   # 60% 缺失 -> 按 max_missing_rate 排除
    "NumE",   # 常量 -> 按常量规则排除
    "NumF",   # 全缺失 -> 按 all-null 规则排除
    "NumG",
    "NumH",
)
CW003_CATEGORICAL = (
    "CatLow",      # 3 类 -> 保留
    "CatConst",    # 单值 -> 按常量规则排除
    "CatHigh",     # 行行不同 -> 按基数上限排除
    "CatMissing",  # 3 类但 60% 缺失 -> 按缺失率排除
)
CW003_IDENTITY_NUMERIC = ("id_num",)
CW003_IDENTITY_CATEGORICAL = ("id_cat",)

N_TIMES = 30
PER_TIME = 10


def build_synthetic_transactions() -> pd.DataFrame:
    """构造覆盖各过滤规则的合成交易表（300 行，30 个时间点各 10 行）。"""
    rng = np.random.default_rng(11)
    rows = N_TIMES * PER_TIME
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(1, rows + 1),
            "TransactionDT": np.repeat(np.arange(100, 100 + N_TIMES), PER_TIME),
        }
    )
    frame["isFraud"] = ((frame.index % 12) == 5).astype(int)
    frame["AmtA"] = np.round(rng.lognormal(3.0, 0.8, rows), 4)
    frame["AmtB"] = rng.integers(0, 50, rows).astype(float)
    frame["NumC"] = rng.normal(size=rows)
    frame["NumD"] = rng.normal(size=rows)
    frame["NumE"] = 1.0
    frame["NumF"] = np.nan
    frame["NumG"] = rng.normal(size=rows)
    frame["NumH"] = rng.integers(-5, 6, rows).astype(float)
    frame["CatLow"] = rng.choice(["a", "b", "c"], rows)
    frame["CatConst"] = "same"
    frame["CatHigh"] = [f"h{index:03d}" for index in range(rows)]
    frame["CatMissing"] = rng.choice(["x", "y", "z"], rows)

    # 缺失注入与标签无关。
    frame.loc[frame.index % 10 == 0, "NumC"] = np.nan
    frame.loc[frame.index % 5 != 0, "NumD"] = np.nan
    frame.loc[frame.index % 5 != 0, "CatMissing"] = np.nan
    return frame


def build_synthetic_identity(
    transaction_ids: np.ndarray, *, coverage: float = 0.6
) -> pd.DataFrame:
    """构造 identity 表。

    ``coverage`` 决定 identity 覆盖多少比例的交易：未覆盖的行在左连接后即为缺失，因此它直接
    决定 identity 列的**真实建模缺失率**。默认 0.6；回归测试会用很低的覆盖率验证
    "identity 列按左连接后的缺失率被排除"。
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage 必须位于 (0, 1]，实际为 {coverage}")
    rng = np.random.default_rng(12)
    covered = transaction_ids[: max(1, int(len(transaction_ids) * coverage))]
    return pd.DataFrame(
        {
            "TransactionID": covered,
            "id_num": rng.normal(size=len(covered)),
            "id_cat": rng.choice(["p", "q"], len(covered)),
        }
    )


class Cw003FixtureBuilder:
    """在临时目录中构造 CW-003 的合成数据、CW-002 产物与配置。"""

    def __init__(self, workspace: Path, *, identity_coverage: float = 0.6,
                 max_cardinality: int = 50) -> None:
        self.workspace = workspace
        #: identity 覆盖率：直接决定 identity 列左连接后的缺失率。
        self.identity_coverage = identity_coverage
        #: 类别基数上限：回归测试会调小它来构造"恰好上限/超过上限"的边界。
        self.max_cardinality = max_cardinality
        self.raw_dir = workspace / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cw002_dir = workspace / "cw002"
        self.cw002_dir.mkdir(parents=True, exist_ok=True)

    # --- 数据 -----------------------------------------------------------

    def write_data(self) -> Path:
        transactions = build_synthetic_transactions()
        identity = build_synthetic_identity(
            transactions["TransactionID"].to_numpy(), coverage=self.identity_coverage
        )
        transactions.to_csv(self.raw_dir / "train_transaction.csv", index=False)
        identity.to_csv(self.raw_dir / "train_identity.csv", index=False)
        return self.raw_dir

    # --- CW-002 产物 -----------------------------------------------------

    def run_cw002_audit(self) -> dict:
        """用 CW-002 的真实审计代码在合成数据上产出清单与审计产物。"""
        from ai_detector.data.audit import run_audit
        from ai_detector.data.ieee_cis import IeeeCisConfig

        report_dir = self.cw002_dir / "reports"
        with (PROJECT_ROOT / "configs" / "ieee_cis.yaml").open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["files"]["raw_dir"] = str(self.raw_dir)
        raw["output"] = {
            "report_dir": str(report_dir),
            "audit_json": str(report_dir / "audit.json"),
            "audit_markdown": str(report_dir / "audit.md"),
            "column_profile_csv": str(report_dir / "column_profile.csv"),
            "split_manifest": str(report_dir / "split_manifest.json"),
            "processed_dir": str(self.workspace / "processed"),
        }
        result = run_audit(IeeeCisConfig.from_mapping(raw))
        return {
            "audit_json": result.audit_json,
            "split_manifest_json": result.split_manifest_json,
            "split_manifest_rows": result.row_partitions_csv,
            "report": result.report,
        }

    # --- CW-003 配置 -----------------------------------------------------

    def build(self) -> tuple[Path, object]:
        """返回 ``(配置路径, ModelingConfig)``。"""
        from ai_detector.data.modeling import ModelingConfig

        self.write_data()
        artifacts = self.run_cw002_audit()
        report = artifacts["report"]

        outputs = {
            "report_dir": str(self.workspace / "out"),
            "report_json": str(self.workspace / "out" / "report.json"),
            "report_markdown": str(self.workspace / "out" / "report.md"),
            "threshold_curve_csv": str(self.workspace / "out" / "threshold_curve.csv"),
            "feature_manifest_json": str(self.workspace / "out" / "feature_manifest.json"),
            "model_dir": str(self.workspace / "models"),
            "dummy_pipeline": str(self.workspace / "models" / "dummy_pipeline.joblib"),
            "logistic_pipeline": str(
                self.workspace / "models" / "logistic_regression_pipeline.joblib"
            ),
            "rule_state": str(self.workspace / "models" / "rule_baseline_state.json"),
        }

        raw = {
            "work_package": "CW-003",
            "seed": 7,
            "dataset": {
                "id": "ieee_cis_fraud_detection",
                "expected_fingerprints": {
                    name: info["sha256"] for name, info in report["files"].items()
                },
                "split_algorithm_version": report["split"]["algorithm_version"],
                "expected_split_sizes": dict(report["split"]["sizes"]),
                "expected_boundaries": dict(report["split"]["boundaries"]),
                "expected_total_rows": report["schema_audit"]["transaction"]["n_rows"],
            },
            "paths": {
                "transaction": str(self.raw_dir / "train_transaction.csv"),
                "identity": str(self.raw_dir / "train_identity.csv"),
                "split_manifest_rows": str(artifacts["split_manifest_rows"]),
                "cw002_audit_json": str(artifacts["audit_json"]),
                "cw002_split_manifest_json": str(artifacts["split_manifest_json"]),
                "outputs": outputs,
            },
            "columns": {
                "target": "isFraud",
                "join_key": "TransactionID",
                "split_time": "TransactionDT",
            },
            "forbidden_features": ["isFraud", "TransactionID", "TransactionDT"],
            "derived": {
                "identity_present": {
                    "enabled": True,
                    "kind": "categorical",
                    "risk_note": "可能携带连接结构信息，属潜在代理变量；已登记风险。",
                }
            },
            "train_only_filters": {
                "max_missing_rate": 0.5,
                "drop_constant": True,
                "drop_all_null": True,
                "categorical": {
                    "max_cardinality": self.max_cardinality,
                    "min_cardinality": 2,
                },
            },
            "preprocessing": {
                "numeric": {
                    "impute_strategy": "median",
                    "add_missing_indicator": True,
                    "scale": "standard",
                    "working_dtype": "float32",
                },
                "categorical": {
                    "impute_strategy": "most_frequent",
                    "encoder": "onehot",
                    "handle_unknown": "ignore",
                    "sparse_output": True,
                    "min_frequency": None,
                },
            },
            "feature_allowlist": {
                "reason_texts": {
                    "anonymous": "匿名统计信号，不赋予业务含义。",
                    "named_but_unconfirmed": "列名有业务暗示但未经公开确认。",
                },
                "sources": {
                    "train_transaction.csv": {
                        "declared_numeric": len(CW003_NUMERIC),
                        "declared_categorical": len(CW003_CATEGORICAL),
                        "declared_total": len(CW003_NUMERIC) + len(CW003_CATEGORICAL),
                        "columns": [
                            {"name": name, "kind": "numeric", "reason_code": "anonymous"}
                            for name in CW003_NUMERIC
                        ]
                        + [
                            {"name": name, "kind": "categorical", "reason_code": "anonymous"}
                            for name in CW003_CATEGORICAL
                        ],
                    },
                    "train_identity.csv": {
                        "declared_numeric": len(CW003_IDENTITY_NUMERIC),
                        "declared_categorical": len(CW003_IDENTITY_CATEGORICAL),
                        "declared_total": len(CW003_IDENTITY_NUMERIC)
                        + len(CW003_IDENTITY_CATEGORICAL),
                        "columns": [
                            {"name": name, "kind": "numeric", "reason_code": "anonymous"}
                            for name in CW003_IDENTITY_NUMERIC
                        ]
                        + [
                            {"name": name, "kind": "categorical", "reason_code": "anonymous"}
                            for name in CW003_IDENTITY_CATEGORICAL
                        ],
                    },
                },
            },
            "rule_baseline": {
                "name": "transparent_tail_anomaly",
                "kind": "unsupervised_tail_heuristic",
                "disclaimer": "研究用启发式，不是业务规则。",
                "fields": [
                    {"name": "AmtA", "rationale": "数值统计信号。"},
                    {"name": "AmtB", "rationale": "数值统计信号。"},
                ],
                "quantile": 0.99,
                "quantile_source": "train",
                "scoring": "mean_of_indicators",
                "score_values": [0.0, 0.5, 1.0],
                "score_range": [0.0, 1.0],
            },
            "models": {
                "dummy": {"type": "dummy_classifier", "params": {"strategy": "prior"}},
                "logistic_regression": {
                    "type": "logistic_regression",
                    "params": {
                        "C": 1.0,
                        "class_weight": "balanced",
                        "solver": "lbfgs",
                        "max_iter": 200,
                        "tol": 0.0001,
                    },
                },
            },
            "thresholds": {
                "grid": {"start": 0.05, "stop": 0.95, "step": 0.05},
                "metric": "f1",
                "f1_tie_break": ["f1_desc", "recall_desc", "threshold_desc"],
                "cost_units": "研究用相对成本单位",
                "cost_disclaimer": "研究假设，非货币。",
                "cost_scenarios": [
                    {"name": "fp_fn_1_1", "fp_cost": 1.0, "fn_cost": 1.0, "rationale": "对称"},
                    {"name": "fp_fn_1_5", "fp_cost": 1.0, "fn_cost": 5.0, "rationale": "偏保守"},
                    {"name": "fp_fn_1_20", "fp_cost": 1.0, "fn_cost": 20.0, "rationale": "强保守"},
                ],
                "cost_tie_break": ["cost_asc", "threshold_desc"],
            },
            "limits": {"evaluation_partition": "test", "single_shot_test": True},
        }

        config_path = self.workspace / "ieee_cis_baseline_test.yaml"
        config_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )
        return config_path, ModelingConfig.from_yaml(config_path)
