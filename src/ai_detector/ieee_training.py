"""CW-003 训练编排：真实 IEEE-CIS 数据上的无泄漏基线实验。

## 执行顺序是硬约束

```text
门禁 -> 复用 CW-002 切分 -> 训练期统计(仅 train) -> 保留列决策(仅 train)
     -> 装配三个分区 -> 拟合规则分位数(仅 train) -> fit 预处理器与模型(仅 train)
     -> 在 validation 上算阈值曲线 -> 冻结 F1 阈值与三个成本阈值
     -> [冻结之后] 在 test 上评估一次 -> 写产物
```

## test 的隔离是结构性的，不是靠约定

- 装配阶段（:func:`assemble_partitions`）就把 test 分区的目标列**剥离并封存**，
  因此冻结前的代码路径**物理上拿不到** test 标签。
- 冻结前只对 test 做结构性检查（连接键集合、分区互斥、不重不漏），见
  :func:`assert_partition_invariants`。
- 所有阈值冻结后调用 :meth:`HoldoutAccessTracker.freeze`，此后才能经追踪器读取 test 标签与
  特征；未冻结即访问会抛 :class:`HoldoutLeakageError`。
- 报告的 ``evaluation.test_access`` 由追踪器在**真实执行阶段**产生，不是事后手工追加的说明。

## 内存

数值列以 ``float32`` 装配；分两遍读取（第一遍只累计统计，第二遍只读保留列）。峰值实测值记入
``WORK_LOG.md``；本机可用内存约 4 GiB。
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_detector import __version__
from ai_detector.data.modeling import (
    ASSEMBLY_CHUNK_SIZE,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    WORK_PACKAGE,
    ModelingConfig,
    ModelingContractError,
    HoldoutAccessTracker,
    assemble_partitions,
    assert_partition_invariants,
    load_partition_lookup,
    validate_test_partition,
    verify_data_gate,
)
from ai_detector.evaluation.cost import (
    cost_matrix_note,
    curve_to_frame,
    metrics_at_threshold,
    select_cost_threshold,
    select_f1_threshold,
    threshold_curve,
)
from ai_detector.evaluation.report import environment_versions, jsonable
from ai_detector.features.ieee_cis import (
    EXCLUSION_REASONS,
    FilterDecision,
    accumulate_train_stats,
    assert_no_forbidden_features,
    assert_no_forbidden_transformed_names,
    build_feature_pipeline,
    decide_columns,
)
from ai_detector.models.ieee_baselines import (
    DUMMY_KEY,
    LOGISTIC_KEY,
    build_estimator,
    describe_estimator,
    fit_with_diagnostics,
)
from ai_detector.models.rule_baseline import RuleBaselineState, fit_rule_baseline

RULE_KEY = "rule_tail_anomaly"


@dataclass
class TrainingResult:
    """一次真实数据基线实验的产物位置与核心结论。"""

    report: dict[str, Any]
    report_json: Path
    report_markdown: Path
    threshold_curve_csv: Path
    feature_manifest_json: Path
    model_paths: dict[str, Path]


@dataclass
class PhaseTimings:
    """各阶段耗时（秒）。"""

    entries: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, seconds: float) -> None:
        self.entries[name] = round(seconds, 3)

    def to_dict(self) -> dict[str, float]:
        return dict(self.entries)


# --- 装配 ---------------------------------------------------------------


def positive_class_scores(estimator: Any, features: Any) -> np.ndarray:
    """取正类概率；估计器若不支持 ``predict_proba`` 则明确失败。"""
    if not hasattr(estimator, "predict_proba"):
        raise ModelingContractError(
            f"{type(estimator).__name__} 不支持 predict_proba，无法生成连续分数"
        )
    return np.asarray(estimator.predict_proba(features))[:, 1]


def freeze_thresholds(
    config: ModelingConfig,
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    *,
    model_key: str,
) -> dict[str, Any]:
    """在 validation 上计算阈值曲线并冻结 F1 阈值与各成本场景阈值。"""
    grid = config.threshold_grid()
    curve = threshold_curve(validation_labels, validation_scores, grid)
    tie_break = [str(value) for value in config.thresholds["f1_tie_break"]]
    cost_tie_break = [str(value) for value in config.thresholds["cost_tie_break"]]

    f1_point = select_f1_threshold(curve, tie_break=tie_break)
    costs: dict[str, Any] = {}
    for scenario in config.cost_scenarios():
        point, total = select_cost_threshold(
            curve,
            fp_cost=float(scenario["fp_cost"]),
            fn_cost=float(scenario["fn_cost"]),
            tie_break=cost_tie_break,
        )
        costs[str(scenario["name"])] = {
            "fp_cost": float(scenario["fp_cost"]),
            "fn_cost": float(scenario["fn_cost"]),
            "selected_threshold": point.threshold,
            "validation_total_cost": total,
            "validation_average_cost": total / point.n_samples if point.n_samples else None,
            "validation_false_positive": point.false_positive,
            "validation_false_negative": point.false_negative,
        }

    return {
        "model_key": model_key,
        "grid_size": len(grid),
        "grid_spec": dict(config.thresholds["grid"]),
        "f1_tie_break": tie_break,
        "cost_tie_break": cost_tie_break,
        "selected_f1_threshold": f1_point.threshold,
        "validation_f1": f1_point.f1,
        "validation_precision": f1_point.precision,
        "validation_recall": f1_point.recall,
        "validation_false_positive": f1_point.false_positive,
        "validation_false_negative": f1_point.false_negative,
        "cost_scenarios": costs,
        "_curve": curve,
        "_f1_point": f1_point,
    }


def evaluate_frozen_thresholds(
    config: ModelingConfig,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    frozen: dict[str, Any],
    *,
    model_key: str,
) -> dict[str, Any]:
    """**冻结之后**在 test 上评估一次。

    本函数是全流程中唯一访问 test 的入口；调用方必须保证它只在所有阈值冻结之后被调用一次。
    """
    metrics = metrics_at_threshold(
        test_labels, test_scores, threshold=float(frozen["selected_f1_threshold"])
    )
    scenarios: dict[str, Any] = {}
    for name, payload in frozen["cost_scenarios"].items():
        point = metrics_at_threshold(
            test_labels, test_scores, threshold=float(payload["selected_threshold"])
        )
        total = point.total_cost(fp_cost=payload["fp_cost"], fn_cost=payload["fn_cost"])
        scenarios[name] = {
            "fp_cost": payload["fp_cost"],
            "fn_cost": payload["fn_cost"],
            "frozen_threshold": payload["selected_threshold"],
            "test_total_cost": total,
            "test_average_cost": total / point.n_samples if point.n_samples else None,
            "test_confusion_matrix": {
                "true_positive": point.true_positive,
                "false_positive": point.false_positive,
                "true_negative": point.true_negative,
                "false_negative": point.false_negative,
            },
        }

    return {
        "model_key": model_key,
        "partition": PARTITION_TEST,
        "frozen_from": PARTITION_VALIDATION,
        "threshold": metrics.threshold,
        "metrics": metrics.to_dict(),
        "cost_scenarios": scenarios,
    }


# --- 报告 ---------------------------------------------------------------


def _render_markdown(report: dict[str, Any]) -> str:
    data = report["data"]
    partitions = report["partitions"]
    baselines = report["baselines"]
    lines = [
        f"# CW-003 真实数据基线报告（{report['work_package']}）",
        "",
        f"- 生成时间（UTC）：{report['generated_at']}",
        f"- 数据集：{report['dataset']['id']}（IEEE-CIS Fraud Detection 公开匿名竞赛数据）",
        f"- 评估分区：{report['evaluation']['evaluation_partition']}，只评估一次",
        "",
        "## 数据门禁与切分",
        "",
        "| 文件 | 相对路径 | 字节数 | SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    for name, info in data["fingerprints"].items():
        lines.append(
            f"| `{name}` | `{info['relative_path']}` | {info['size_bytes']:,} | `{info['sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"- 切分来源：CW-002 行级清单（算法 `{data['split_algorithm_version']}`），"
            "本工作包不重新推导边界、不做随机切分。",
            f"- 边界：`validation_start = {report['dataset']['boundaries']['validation_start']}`，"
            f"`test_start = {report['dataset']['boundaries']['test_start']}`（原始 TransactionDT 数值）",
            f"- 配置 SHA-256：`{report['config']['config_sha256']}`",
            "",
            "| 分区 | 行数 | 标签率 |",
            "| --- | --- | --- |",
        ]
    )
    for name in (PARTITION_TRAIN, PARTITION_VALIDATION, PARTITION_TEST):
        lines.append(
            f"| {name} | {partitions[name]['rows']:,} | {partitions[name]['label_rate']:.4%} |"
        )

    lines.extend(
        [
            "",
            "## 特征契约",
            "",
            f"- 允许清单：transaction {report['features']['allowlist']['train_transaction.csv']} 列、"
            f"identity {report['features']['allowlist']['train_identity.csv']} 列（显式枚举，见配置文件）。",
            f"- 训练期排除：{report['features']['decision']['excluded_total']} 列"
            "（按训练分区缺失率、常量性与类别基数规则，阈值在配置中预先声明）。",
            f"- 最终特征：{report['features']['decision']['kept_total']} 列，"
            f"转换后维度 {report['features']['transformed_dimension']}，"
            f"输出格式 {report['features']['output_format']}。",
            "- 目标列、连接键与原始时间列在所有阶段通过硬校验排除。",
            "",
            "## 三类基线",
            "",
            "| 基线 | 拟合耗时(s) | 推理耗时(s) | 收敛 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in (RULE_KEY, DUMMY_KEY, LOGISTIC_KEY):
        entry = baselines[key]
        converged = entry["convergence"]["converged"]
        converged_text = "不适用" if converged is None else str(converged)
        lines.append(
            f"| {key} | {entry['fit_seconds']} | {entry['predict_seconds']} | {converged_text} |"
        )

    lines.extend(
        [
            "",
            "## 验证集冻结的阈值",
            "",
            "| 基线 | F1 阈值 | 验证 F1 | 相对成本场景阈值 (1:1 / 1:5 / 1:20) |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in (RULE_KEY, DUMMY_KEY, LOGISTIC_KEY):
        frozen = baselines[key]["thresholds"]
        scenario_thresholds = " / ".join(
            f"{frozen['cost_scenarios'][name]['selected_threshold']:.2f}"
            for name in ("fp_fn_1_1", "fp_fn_1_5", "fp_fn_1_20")
        )
        lines.append(
            f"| {key} | {frozen['selected_f1_threshold']:.2f} | {frozen['validation_f1']:.4f} | "
            f"{scenario_thresholds} |"
        )

    lines.extend(
        [
            "",
            "## 测试集一次性评估（阈值冻结之后）",
            "",
            "| 基线 | 阈值 | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | 预测正例率 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for key in (RULE_KEY, DUMMY_KEY, LOGISTIC_KEY):
        metrics = baselines[key]["test"]["metrics"]
        lines.append(
            f"| {key} | {metrics['threshold']:.2f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} | "
            f"{metrics['pr_auc']:.4f} | {metrics['false_positive_rate']:.4f} | "
            f"{metrics['predicted_positive_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### 相对成本结果（研究用相对单位，非货币）",
            "",
            "| 基线 | 场景 | FP:FN | 冻结阈值 | 测试总成本 | 每笔平均成本 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for key in (RULE_KEY, DUMMY_KEY, LOGISTIC_KEY):
        for name, payload in baselines[key]["test"]["cost_scenarios"].items():
            lines.append(
                f"| {key} | {name} | {payload['fp_cost']:.0f}:{payload['fn_cost']:.0f} | "
                f"{payload['frozen_threshold']:.2f} | {payload['test_total_cost']:,.0f} | "
                f"{payload['test_average_cost']:.6f} |"
            )

    lines.extend(["", "## 成本口径与限制", ""])
    for key, value in report["cost_note"].items():
        lines.append(f"- **{key}**：{value}")
    lines.extend(["", "## 风险与限制", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 环境版本", "", "| 组件 | 版本 |", "| --- | --- |"])
    for key, value in report["environment"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "> 本报告基于公开匿名竞赛数据，**不是**生产风控系统，也不代表任何真实银行的线上效果。"
            "报告未设置任何「效果必须达到某值」的门槛。",
            "",
        ]
    )
    return "\n".join(lines)


# --- 主流程 -------------------------------------------------------------


def run_training(
    config: ModelingConfig,
    *,
    chunk_size: int = ASSEMBLY_CHUNK_SIZE,
    measure: bool = True,
) -> TrainingResult:
    """执行完整实验。

    Raises:
        DataGateError: 开工门禁不通过（应据此置为 blocked）。
        ModelingContractError: 数据契约或不变量被破坏。
    """
    started = time.perf_counter()
    timings = PhaseTimings()

    phase = time.perf_counter()
    gate = verify_data_gate(config)
    timings.record("data_gate", time.perf_counter() - phase)

    phase = time.perf_counter()
    lookup = load_partition_lookup(config)
    timings.record("load_split_manifest", time.perf_counter() - phase)

    phase = time.perf_counter()
    stats = accumulate_train_stats(config, lookup, chunk_size=chunk_size)
    decision = decide_columns(config, stats)
    timings.record("train_only_column_decisions", time.perf_counter() - phase)

    phase = time.perf_counter()
    assembled = assemble_partitions(config, lookup, decision, chunk_size=chunk_size)
    # 冻结前只做结构性检查（连接键集合、互斥、不重不漏）；**不读 test 标签**。
    invariants = assert_partition_invariants(config, assembled)
    timings.record("assemble_partitions", time.perf_counter() - phase)
    del lookup
    gc.collect()

    # test 访问追踪器：冻结前访问 test 标签/特征会直接抛 HoldoutLeakageError。
    tracker = HoldoutAccessTracker()

    train_frame = assembled[PARTITION_TRAIN]
    validation_frame = assembled[PARTITION_VALIDATION]

    train_target = assembled.target(PARTITION_TRAIN, tracker=tracker)
    validation_target = assembled.target(PARTITION_VALIDATION, tracker=tracker)

    rule_state, rule_train_seconds = _fit_rule(
        config, train_frame, timings
    )

    frozen: dict[str, dict[str, Any]] = {}
    baselines: dict[str, Any] = {}

    # --- 规则基线 ---
    validation_rule_scores = rule_state.score(validation_frame)
    phase = time.perf_counter()
    rule_frozen = freeze_thresholds(
        config, validation_target, validation_rule_scores, model_key=RULE_KEY
    )
    timings.record("rule_validation_curve", time.perf_counter() - phase)
    frozen[RULE_KEY] = rule_frozen

    # --- sklearn 基线（在 train 上 fit 预处理与模型）---
    for key, model_type in ((DUMMY_KEY, "dummy_classifier"), (LOGISTIC_KEY, "logistic_regression")):
        spec = config.models[key]
        estimator = build_estimator(
            str(spec["type"]), dict(spec.get("params") or {}), seed=config.seed
        )
        pipeline = build_feature_pipeline(config, decision, estimator)

        feature_columns = _feature_columns_for(config, decision)
        assert_no_forbidden_features(feature_columns, config.forbidden_features)

        phase = time.perf_counter()
        diagnostics = fit_with_diagnostics(
            pipeline,
            assembled.features(PARTITION_TRAIN, feature_columns),
            assembled.target(PARTITION_TRAIN, tracker=tracker),
            # n_iter_ 只存在于最终估计器上，必须显式传入，否则无法判断是否收敛。
            diagnostics_estimator=estimator,
            # 配置中声明的收敛相关调整理由会原样进入报告，供验收者核对。
            adjustment_note=spec.get("adjustment_note"),
        )
        fit_seconds = time.perf_counter() - phase
        timings.record(f"{key}_fit", fit_seconds)

        phase = time.perf_counter()
        validation_scores = positive_class_scores(
            pipeline, assembled.features(PARTITION_VALIDATION, feature_columns)
        )
        predict_seconds = time.perf_counter() - phase
        timings.record(f"{key}_predict_validation", predict_seconds)

        model_frozen = freeze_thresholds(
            config, validation_target, validation_scores, model_key=key
        )
        frozen[key] = model_frozen

        baselines[key] = {
            "type": model_type,
            "params": describe_estimator(estimator),
            "fit_seconds": round(fit_seconds, 3),
            "predict_seconds": round(predict_seconds, 3),
            "convergence": diagnostics.to_dict(),
            "thresholds": {k: v for k, v in model_frozen.items() if not k.startswith("_")},
            "_pipeline": pipeline,
            "_validation_scores": validation_scores,
        }
        del validation_scores
        gc.collect()

    # 两个 sklearn 基线都已在 train 上拟合完毕，此后不再需要训练分区的原始数据。
    del train_frame
    assembled.release(PARTITION_TRAIN)
    gc.collect()

    del validation_frame
    assembled.release(PARTITION_VALIDATION)
    gc.collect()

    # --- 所有阈值冻结完毕；此后才允许访问 test，且只访问一次 ---
    phase = time.perf_counter()
    tracker.freeze(
        reason=(
            "全部基线的 F1 阈值与成本场景阈值均已在 validation 上选出并冻结；"
            "此后进入受控的 test 一次性评估阶段"
        )
    )
    test_partition_summary = validate_test_partition(config, assembled, tracker)
    baselines[RULE_KEY] = {
        "type": "rule_baseline",
        "params": {"fields": list(rule_state.fields), "quantile": rule_state.quantile,
                   "thresholds": dict(rule_state.thresholds), "scoring": rule_state.scoring},
        "fit_seconds": rule_train_seconds,
        "predict_seconds": 0.0,
        "convergence": {"estimator": "RuleBaseline", "converged": None, "n_iter": [],
                        "warnings": [], "adjustments": []},
        "thresholds": {k: v for k, v in frozen[RULE_KEY].items() if not k.startswith("_")},
        "_state": rule_state,
    }
    # test 特征只能经受控入口取得：冻结检查与访问记账由 features() 自己完成，
    # 且发生在返回任何数据之前；调用方无法"先拿数据、再补日志"。
    test_features = assembled.features(
        PARTITION_TEST, _feature_columns_for(config, decision), tracker=tracker
    )
    test_target = assembled.target(PARTITION_TEST, tracker=tracker)
    # 规则基线只读它声明的字段，这些字段都在 test_features 中，因此复用同一份受控数据，
    # 不再单独取出 test 整帧。
    test_scores = {RULE_KEY: rule_state.score(test_features)}
    for key in (DUMMY_KEY, LOGISTIC_KEY):
        test_scores[key] = positive_class_scores(
            baselines[key]["_pipeline"], test_features
        )

    for key in (RULE_KEY, DUMMY_KEY, LOGISTIC_KEY):
        baselines[key]["test"] = {
            **_evaluate_with_auc(
                config, test_target, test_scores[key], frozen[key], model_key=key
            )
        }
    timings.record("frozen_test_evaluation", time.perf_counter() - phase)
    del test_features
    assembled.release(PARTITION_TEST)
    gc.collect()

    transformed_dimension, output_format = _describe_transform(
        baselines.get(LOGISTIC_KEY, {}).get("_pipeline")
    )
    feature_manifest = _build_feature_manifest(
        config, decision, transformed_dimension, output_format
    )

    report = _build_report(
        config=config,
        gate=gate,
        invariants=invariants,
        decision=decision,
        baselines=baselines,
        feature_manifest=feature_manifest,
        timings=timings,
        total_seconds=time.perf_counter() - started,
        test_access=tracker.to_dict(),
        test_partition_summary=test_partition_summary,
        transformed_dimension=transformed_dimension,
        output_format=output_format,
    )

    outputs = _write_artifacts(config, report, baselines, feature_manifest, frozen)
    return TrainingResult(report=report, **outputs)


def _feature_columns_for(config: ModelingConfig, decision: FilterDecision) -> list[str]:
    columns = list(decision.kept)
    if bool(config.derived.get("identity_present", {}).get("enabled")):
        columns.append("identity_present")
    return columns


def _fit_rule(
    config: ModelingConfig, train_frame: pd.DataFrame, timings: PhaseTimings
) -> tuple[RuleBaselineState, float]:
    phase = time.perf_counter()
    state = fit_rule_baseline(config.rule_baseline, train_frame)
    seconds = time.perf_counter() - phase
    timings.record("rule_baseline_fit", seconds)
    return state, round(seconds, 3)


def _evaluate_with_auc(
    config: ModelingConfig,
    test_target: np.ndarray,
    test_scores: np.ndarray,
    frozen: dict[str, Any],
    *,
    model_key: str,
) -> dict[str, Any]:
    """冻结阈值下的测试评估，并补齐 ROC-AUC / PR-AUC。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    payload = evaluate_frozen_thresholds(
        config, test_target, test_scores, frozen, model_key=model_key
    )
    if np.unique(test_target).size < 2:
        raise ModelingContractError("测试分区只有单一类别，ROC-AUC / PR-AUC 无定义")
    payload["metrics"]["roc_auc"] = float(roc_auc_score(test_target, test_scores))
    payload["metrics"]["pr_auc"] = float(average_precision_score(test_target, test_scores))
    return payload


def _describe_transform(pipeline: Any) -> tuple[int, str]:
    if pipeline is None:
        return 0, "unknown"
    preprocessor = pipeline.named_steps["preprocessor"]
    names = list(preprocessor.get_feature_names_out())
    assert_no_forbidden_transformed_names(
        names, ("isFraud", "TransactionID", "TransactionDT")
    )
    return len(names), "sparse_csr"


def _build_feature_manifest(
    config: ModelingConfig,
    decision: FilterDecision,
    transformed_dimension: int,
    output_format: str,
) -> dict[str, Any]:
    return {
        "work_package": WORK_PACKAGE,
        "reason_texts": dict(config.reason_texts),
        "allowlist": {
            source: [entry["name"] for entry in entries]
            for source, entries in config.allowlist.items()
        },
        "kept": {"numeric": decision.numeric, "categorical": decision.categorical},
        "excluded": decision.excluded,
        "train_stats": {name: stats.to_dict() for name, stats in decision.stats.items()},
        "transformed_dimension": transformed_dimension,
        "output_format": output_format,
        "forbidden_features": list(config.forbidden_features),
        "derived_features": {
            name: {"enabled": bool(payload.get("enabled")), "risk_note": payload.get("risk_note", "")}
            for name, payload in config.derived.items()
        },
        "notes": [
            "训练期统计只来自 train 分区；validation/test 只应用已冻结的决策。",
            "类别列不输出原始取值，只输出计数类摘要。",
            "匿名列只登记为统计信号，未赋予任何业务含义。",
        ],
    }


def _build_report(
    *,
    config: ModelingConfig,
    gate: dict[str, Any],
    invariants: dict[str, Any],
    decision: FilterDecision,
    baselines: dict[str, Any],
    feature_manifest: dict[str, Any],
    timings: PhaseTimings,
    total_seconds: float,
    test_access: dict[str, Any],
    test_partition_summary: dict[str, Any],
    transformed_dimension: int,
    output_format: str,
) -> dict[str, Any]:
    clean_baselines: dict[str, Any] = {}
    for key, payload in baselines.items():
        clean_baselines[key] = {k: v for k, v in payload.items() if not k.startswith("_")}

    return {
        "work_package": WORK_PACKAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "seed": config.seed,
        "dataset": {
            "id": config.dataset_id,
            "boundaries": gate["boundaries"],
            "expected_total_rows": config.expected_total_rows,
        },
        "config": {
            "relative_path": config.relative(config.config_path),
            "config_sha256": config.config_sha256,
        },
        "data": {
            "fingerprints": gate["fingerprints"],
            "split_manifest": gate["split_manifest"],
            "split_algorithm_version": gate["split_algorithm_version"],
        },
        "partitions": {
            PARTITION_TRAIN: {
                "rows": invariants["train_rows"],
                "label_rate": invariants["label_rates"][PARTITION_TRAIN],
            },
            PARTITION_VALIDATION: {
                "rows": invariants["validation_rows"],
                "label_rate": invariants["label_rates"][PARTITION_VALIDATION],
            },
            # test 的标签率来自冻结后的受控阶段，不在冻结前计算。
            PARTITION_TEST: {
                "rows": invariants["test_rows"],
                "label_rate": test_partition_summary["label_rate"],
                "positive_count": test_partition_summary["positive_count"],
                "label_rate_computed_after_freeze": True,
            },
            "invariants": {
                "ids_disjoint": invariants["ids_disjoint"],
                "ids_complete": invariants["ids_complete"],
            },
        },
        "features": {
            "allowlist": {
                source: len(entries) for source, entries in config.allowlist.items()
            },
            "decision": decision.to_dict(),
            "excluded_breakdown": _excluded_breakdown(decision),
            "transformed_dimension": transformed_dimension,
            "output_format": output_format,
            "forbidden_features": list(config.forbidden_features),
        },
        "baselines": clean_baselines,
        "risk_register": [
            {
                "item": "identity_present",
                "risk": "编码是否存在 identity 记录，可能携带连接结构或采集差异信息，属潜在代理变量。",
                "status": "已登记并作为显式特征使用；未声称已排除风险。",
            },
            {
                "item": "匿名列（V*/C*/D*/M*/id_*）",
                "risk": "语义不可从列名确认，可能包含未知代理变量或时间邻近泄漏。",
                "status": "已登记；本轮只做审计与基线，不声称已排除。",
            },
            {
                "item": "高基数类别列",
                "risk": "可能近似唯一标识，引入过拟合或身份泄漏。",
                "status": "已由训练期基数上限规则排除（见 excluded_breakdown）。",
            },
        ],
        "cost_note": cost_matrix_note(),
        "evaluation": {
            "evaluation_partition": config.limits.get("evaluation_partition", PARTITION_TEST),
            "single_shot_test": bool(config.limits.get("single_shot_test", True)),
            "validation_used_for": ["threshold_curve", "f1_threshold", "cost_thresholds"],
            "test_used_for": ["single_frozen_evaluation"],
            # 由 HoldoutAccessTracker 在真实执行阶段产生，不是事后手工追加的说明。
            "test_access": test_access,
            "pre_freeze_structural_only": {
                "description": (
                    "阈值冻结之前只对 test 做结构性检查（连接键集合、分区互斥、不重不漏）；"
                    "test 标签在装配阶段即被封存，冻结前不可读"
                ),
                "test_labels_read_before_freeze": invariants["test_labels_read_before_freeze"],
                "label_rates_computed_for": invariants["label_rates_computed_for"],
            },
        },
        "timings_seconds": timings.to_dict(),
        "total_seconds": round(total_seconds, 3),
        "limitations": _limitations(config),
        "environment": environment_versions(),
    }


def _excluded_breakdown(decision: FilterDecision) -> dict[str, int]:
    breakdown: dict[str, int] = {code: 0 for code in EXCLUSION_REASONS}
    for item in decision.excluded:
        breakdown[item["reason_code"]] = breakdown.get(item["reason_code"], 0) + 1
    return breakdown


def _limitations(config: ModelingConfig) -> list[str]:
    return [
        "匿名列是否包含未知代理或泄漏风险未解决；本轮只登记风险，不声称已排除。",
        "高基数类别列被训练期基数上限排除（阈值预先声明），因此部分有信息的列未进入模型。",
        "数值列统一按数值处理，未区分「数值编码的类别」与「连续量」；这是建模简化，不是语义断言。",
        "缺失指示器按训练期缺失情况添加，宽度由配置阈值约束；未做缺失模式的专门建模。",
        "类别列不输出原始取值，报告无法直接查看类别分布。",
        "相对成本比例是研究假设，不是真实货币损失，也不代表任何机构的风险偏好。",
        "文字中的「金额」等措辞只是列名的字面转述，未经公开确认；不对匿名列的真实含义作断言。",
        "IEEE-CIS 是匿名化的公开竞赛数据，不能视为真实银行生产数据；报告不设置任何效果门槛，"
        "也不将结果外推为生产效果。",
        "Logistic Regression 是线性模型，未做特征交互或非线性变换；这不是效果声明。",
    ]


def _write_artifacts(
    config: ModelingConfig,
    report: dict[str, Any],
    baselines: dict[str, Any],
    feature_manifest: dict[str, Any],
    frozen: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    import joblib

    outputs = config.outputs
    outputs["report_dir"].mkdir(parents=True, exist_ok=True)
    outputs["model_dir"].mkdir(parents=True, exist_ok=True)

    outputs["report_json"].write_text(
        json_dumps(report), encoding="utf-8"
    )
    outputs["report_markdown"].write_text(_render_markdown(report), encoding="utf-8")
    outputs["feature_manifest_json"].write_text(
        json_dumps(feature_manifest), encoding="utf-8"
    )

    curve_frames = []
    for key, payload in frozen.items():
        frame = curve_to_frame(payload["_curve"])
        frame.insert(0, "model_key", key)
        frame.insert(1, "partition", PARTITION_VALIDATION)
        curve_frames.append(frame)
    pd.concat(curve_frames, ignore_index=True).to_csv(
        outputs["threshold_curve_csv"], index=False
    )

    model_paths: dict[str, Path] = {}
    for key in (DUMMY_KEY, LOGISTIC_KEY):
        path = outputs[f"{'dummy' if key == DUMMY_KEY else 'logistic'}_pipeline"]
        joblib.dump(baselines[key]["_pipeline"], path)
        model_paths[key] = path

    rule_state: RuleBaselineState = baselines[RULE_KEY]["_state"]
    rule_state.write(outputs["rule_state"])
    model_paths[RULE_KEY] = outputs["rule_state"]

    return {
        "report_json": outputs["report_json"],
        "report_markdown": outputs["report_markdown"],
        "threshold_curve_csv": outputs["threshold_curve_csv"],
        "feature_manifest_json": outputs["feature_manifest_json"],
        "model_paths": model_paths,
    }


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n"
