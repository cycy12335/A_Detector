"""test 分区读取门禁的定向单元测试（不依赖真实数据，秒级完成）。

首轮独立验收的阻塞项是：``AssembledData.features("test", ...)`` 在不冻结的情况下也能成功
返回数据，且不产生任何日志——因此"test 只读一次"的报告只是手工记账，不是结构性保证。
本文件把该缺陷的每一个逃逸路径都钉成回归测试：

1. 冻结前读 test 特征必须抛错，且**不返回数据、不留下日志**；
2. 不提供追踪器时拒绝返回 test 特征（否则就没有冻结检查与记账）；
3. ``assembled["test"]`` 整体取出被拒绝（第二条未受保护的读取路径）；
4. 冻结后恰好一次成功读取，且日志由读取入口自身产生；
5. train / validation 完全不受影响；
6. 只读元信息（列名、行数、连接键）始终可用，但只暴露 schema 与结构。
"""

from __future__ import annotations

import pandas as pd
import pytest

from ai_detector.data.modeling import (
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    AssembledData,
    HoldoutAccessTracker,
    HoldoutLeakageError,
    ModelingContractError,
)

TARGET = "isFraud"
JOIN_KEY = "TransactionID"
COLUMNS = ["AmtA", "NumB"]


def _assembled() -> AssembledData:
    """构造一个最小的装配结果：test 的目标列已剥离并封存。"""
    train = pd.DataFrame(
        {JOIN_KEY: [1, 2, 3], TARGET: [0, 1, 0], "AmtA": [1.0, 2.0, 3.0], "NumB": [4.0, 5.0, 6.0]}
    )
    validation = pd.DataFrame(
        {JOIN_KEY: [4, 5], TARGET: [0, 0], "AmtA": [7.0, 8.0], "NumB": [9.0, 10.0]}
    )
    test = pd.DataFrame({JOIN_KEY: [6, 7], "AmtA": [11.0, 12.0], "NumB": [13.0, 14.0]})
    return AssembledData(
        _frames={PARTITION_TRAIN: train, PARTITION_VALIDATION: validation, PARTITION_TEST: test},
        target_column=TARGET,
        join_key_column=JOIN_KEY,
        _sealed_test_target=pd.Series([0, 1]).to_numpy(),
    )


# --- 冻结前：必须失败且不留痕 -------------------------------------------


def test_pre_freeze_test_features_raise_and_return_nothing() -> None:
    assembled = _assembled()
    tracker = HoldoutAccessTracker()

    with pytest.raises(HoldoutLeakageError, match="在阈值冻结之前尝试访问 test"):
        assembled.features(PARTITION_TEST, COLUMNS, tracker=tracker)
    assert tracker.entries == []
    assert tracker.frozen is False


def test_pre_freeze_test_features_raise_without_a_tracker_too() -> None:
    assembled = _assembled()
    with pytest.raises(HoldoutLeakageError, match="必须提供生命周期追踪器"):
        assembled.features(PARTITION_TEST, COLUMNS)


def test_tracker_absence_is_rejected_even_after_someone_freezes_their_own() -> None:
    """别人的追踪器冻结了，不等于这次读取被记账——因此这里仍然拒绝。"""
    assembled = _assembled()
    HoldoutAccessTracker().freeze(reason="另一个互不相干的追踪器")

    with pytest.raises(HoldoutLeakageError, match="必须提供生命周期追踪器"):
        assembled.features(PARTITION_TEST, COLUMNS)


def test_whole_test_frame_is_rejected() -> None:
    assembled = _assembled()
    with pytest.raises(HoldoutLeakageError, match="不得经 assembled\\['test'\\] 整体取出"):
        assembled[PARTITION_TEST]


def test_pre_freeze_test_labels_raise_and_log_nothing() -> None:
    assembled = _assembled()
    tracker = HoldoutAccessTracker()

    with pytest.raises(HoldoutLeakageError, match="在阈值冻结之前尝试访问 test"):
        assembled.target(PARTITION_TEST, tracker=tracker)
    assert tracker.entries == []


# --- 冻结后：恰好一次，且由入口自己记账 ----------------------------------


def test_post_freeze_features_read_once_and_log_once() -> None:
    assembled = _assembled()
    tracker = HoldoutAccessTracker()
    tracker.freeze(reason="单元测试")

    frame = assembled.features(PARTITION_TEST, COLUMNS, tracker=tracker)

    assert list(frame.columns) == COLUMNS
    assert len(frame) == 2
    accesses = [entry for entry in tracker.entries if entry["event"] == "test_access"]
    assert len(accesses) == 1
    assert accesses[0]["what"] == "features"
    assert accesses[0]["purpose"] == "single_shot_evaluation"
    assert accesses[0]["phase"] == "post_freeze_evaluation"
    assert accesses[0]["after_freeze"] is True


def test_post_freeze_labels_read_once_and_log_once() -> None:
    assembled = _assembled()
    tracker = HoldoutAccessTracker()
    tracker.freeze(reason="单元测试")

    labels = assembled.target(PARTITION_TEST, tracker=tracker)

    assert labels.tolist() == [0, 1]
    accesses = [entry for entry in tracker.entries if entry["event"] == "test_access"]
    assert [entry["what"] for entry in accesses] == ["labels"]


def test_repeated_reads_produce_one_entry_each() -> None:
    """日志条数等于真实读取次数：不存在"读了但没记"或"没读却记了"。"""
    assembled = _assembled()
    tracker = HoldoutAccessTracker()
    tracker.freeze(reason="单元测试")

    assembled.features(PARTITION_TEST, COLUMNS, tracker=tracker)
    assembled.features(PARTITION_TEST, COLUMNS, tracker=tracker)
    assembled.target(PARTITION_TEST, tracker=tracker)

    accesses = [entry["what"] for entry in tracker.entries if entry["event"] == "test_access"]
    assert accesses == ["features", "features", "labels"]


def test_first_entry_is_the_freeze_event() -> None:
    assembled = _assembled()
    tracker = HoldoutAccessTracker()
    tracker.freeze(reason="单元测试：全部阈值已冻结")
    assembled.features(PARTITION_TEST, COLUMNS, tracker=tracker)

    assert tracker.entries[0]["event"] == "thresholds_frozen"
    assert tracker.entries[0]["reason"] == "单元测试：全部阈值已冻结"


# --- train / validation 不受影响 ----------------------------------------


def test_train_and_validation_need_no_tracker_and_log_nothing() -> None:
    assembled = _assembled()
    tracker = HoldoutAccessTracker()

    assert len(assembled.features(PARTITION_TRAIN, COLUMNS)) == 3
    assert len(assembled.features(PARTITION_VALIDATION, COLUMNS)) == 2
    assert assembled[PARTITION_TRAIN] is not None
    assert assembled[PARTITION_VALIDATION] is not None
    assert assembled.target(PARTITION_TRAIN, tracker=tracker).tolist() == [0, 1, 0]
    assert tracker.entries == []


def test_missing_columns_still_raise_contract_error() -> None:
    assembled = _assembled()
    with pytest.raises(ModelingContractError, match="缺少特征列"):
        assembled.features(PARTITION_TRAIN, ["AmtA", "NopeNotAColumn"])


# --- 只读元信息 ---------------------------------------------------------


def test_structural_accessors_expose_no_values() -> None:
    assembled = _assembled()

    assert assembled.structural_row_count(PARTITION_TEST) == 2
    assert assembled.structural_columns(PARTITION_TEST) == [JOIN_KEY, *COLUMNS]
    assert assembled.structural_ids(PARTITION_TEST) == {6, 7}
    # train/validation 仍带目标列；test 在装配阶段就被剥离并封存。
    assert TARGET in assembled.structural_columns(PARTITION_TRAIN)
    assert TARGET in assembled.structural_columns(PARTITION_VALIDATION)
    assert TARGET not in assembled.structural_columns(PARTITION_TEST)


def test_release_drops_the_frame_without_affecting_others() -> None:
    assembled = _assembled()
    assembled.release(PARTITION_TEST)

    assert PARTITION_TEST not in assembled._frames
    assert assembled.structural_row_count(PARTITION_TRAIN) == 3
