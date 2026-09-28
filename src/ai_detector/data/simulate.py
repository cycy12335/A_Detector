"""确定性模拟交易数据生成器。

**这是模拟数据，不是真实交易数据。**

生成逻辑是一个显式的混合模型（mixture）：每笔交易先抽一个不可见的
``risky`` 隐变量，再由它同时影响特征分布和欺诈概率。因此欺诈信号是可以被
模型学到的，但没有任何输出列等于标签本身或标签的确定性函数。

局限（必须随结果一起说明）：

- 分布形状、系数和基础欺诈率都是人为设定的，不代表任何真实机构的数据。
- 三个特征由同一个隐变量 ``risky`` 条件生成，因此在边际上**并不独立**：``risky``
  会同时推高 ``amount``、``transaction_count_1h`` 与 ``is_new_device``。这是一个
  简化的条件生成过程，缺少真实用户行为序列、商户网络和长期时序依赖；不要把它当作
  独立特征生成器或真实数据的近似。
- 因此本数据上的任何指标只能用于验证工程闭环，不能外推到真实风控效果。
- 用户级隐风险 ``user_risk`` 会影响标签但不进入特征，用于保留一部分不可学习的
  噪声；这使模型不可能接近完美，避免产生误导性的漂亮指标。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ai_detector.data.schema import (
    ALL_COLUMNS,
    NUMERIC_FEATURES,
    TIMESTAMP_COLUMN,
    TRANSACTION_ID_COLUMN,
    validate_transactions,
)

DEFAULT_SEED = 42
DEFAULT_ROWS = 5000
DEFAULT_START_TIMESTAMP = "2025-01-01T00:00:00"
DEFAULT_MISSING_RATE = 0.02

# --- 生成模型系数（按业务直觉设定，未依据任何模型指标调参） ---
_P_RISKY = 0.12
_BASE_LOGIT = -3.6
_RISKY_LOGIT = 2.2
_USER_RISK_LOGIT = 0.6
_P_NEW_DEVICE_RISKY = 0.55
_P_NEW_DEVICE_NORMAL = 0.06
_AMOUNT_LOG_MEAN = 3.2
_AMOUNT_LOG_SIGMA = 1.0
_AMOUNT_RISKY_MULTIPLIER = 0.9
_COUNT_LAMBDA_RISKY = 9.0
_COUNT_LAMBDA_NORMAL = 3.0
_USER_RISK_BETA_A = 2.0
_USER_RISK_BETA_B = 8.0
_MAX_DEVICES_PER_USER = 3
_MIN_USERS = 50
_ROWS_PER_USER = 12


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_transactions(
    *,
    seed: int = DEFAULT_SEED,
    rows: int = DEFAULT_ROWS,
    start_timestamp: str = DEFAULT_START_TIMESTAMP,
    missing_rate: float = DEFAULT_MISSING_RATE,
) -> pd.DataFrame:
    """生成按时间升序的模拟交易数据。

    相同参数必然产生完全相同的内容：所有随机性都来自 ``seed`` 初始化的
    ``numpy.random.Generator``，生成过程不读取时钟、环境变量或外部文件。

    Args:
        seed: 随机种子。
        rows: 生成行数，必须为正整数。
        start_timestamp: 首笔交易时间（ISO 8601 字符串）。
        missing_rate: 注入到数值特征上的缺失比例，用于让缺失值处理路径真正被执行。
            缺失注入与标签无关。取值必须在 ``[0, 1)``。

    Returns:
        包含 :data:`ai_detector.data.schema.ALL_COLUMNS` 各列的数据框，
        已按 ``timestamp`` 升序排列。
    """
    if rows < 1:
        raise ValueError(f"rows 必须为正整数，实际为 {rows}")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(f"missing_rate 必须位于 [0, 1)，实际为 {missing_rate}")

    rng = np.random.default_rng(seed)

    n_users = max(_MIN_USERS, rows // _ROWS_PER_USER)
    raw_user_risk = rng.beta(_USER_RISK_BETA_A, _USER_RISK_BETA_B, size=n_users)
    user_risk = (raw_user_risk - raw_user_risk.mean()) / raw_user_risk.std()
    device_pool_size = rng.integers(1, _MAX_DEVICES_PER_USER + 1, size=n_users)

    user_index = rng.integers(0, n_users, size=rows)
    risky = rng.random(rows) < _P_RISKY

    probability_new_device = np.where(risky, _P_NEW_DEVICE_RISKY, _P_NEW_DEVICE_NORMAL)
    is_new_device = (rng.random(rows) < probability_new_device).astype("int64")

    # 已知设备只能落在该用户已拥有的设备池内；新设备使用池外的下标，
    # 从而保证 device_id 与 is_new_device 语义一致。
    device_index = rng.integers(0, _MAX_DEVICES_PER_USER, size=rows)
    device_index = np.minimum(device_index, device_pool_size[user_index] - 1)
    device_index = np.where(is_new_device == 1, device_pool_size[user_index], device_index)
    device_ids = [
        f"D{user:05d}-{device:02d}" for user, device in zip(user_index, device_index, strict=True)
    ]

    amount = rng.lognormal(
        mean=_AMOUNT_LOG_MEAN, sigma=_AMOUNT_LOG_SIGMA, size=rows
    ) * np.where(risky, np.exp(_AMOUNT_RISKY_MULTIPLIER), 1.0)
    amount = np.round(amount, 2)

    transaction_count_1h = rng.poisson(
        np.where(risky, _COUNT_LAMBDA_RISKY, _COUNT_LAMBDA_NORMAL)
    ).astype("float64")

    # 严格递增的时间戳：分钟级间隔至少为 1，保证三个时间区间之间不会出现并列时间。
    gaps_minutes = rng.integers(1, 8, size=rows)
    offsets = np.cumsum(gaps_minutes)
    timestamps = pd.Timestamp(start_timestamp) + pd.to_timedelta(offsets, unit="m")

    fraud_logit = (
        _BASE_LOGIT + _RISKY_LOGIT * risky + _USER_RISK_LOGIT * user_risk[user_index]
    )
    is_fraud = (rng.random(rows) < _sigmoid(fraud_logit)).astype("int64")

    frame = pd.DataFrame(
        {
            "user_id": [f"U{index:05d}" for index in user_index],
            "device_id": device_ids,
            "amount": amount,
            "transaction_count_1h": transaction_count_1h,
            "is_new_device": is_new_device,
            "is_fraud": is_fraud,
            TIMESTAMP_COLUMN: timestamps,
        }
    )

    if missing_rate > 0:
        n_missing = int(round(rows * missing_rate))
        for column in NUMERIC_FEATURES:
            missing_positions = rng.choice(rows, size=n_missing, replace=False)
            frame.loc[missing_positions, column] = np.nan

    frame = frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    frame.insert(0, TRANSACTION_ID_COLUMN, [f"TX{i:08d}" for i in range(len(frame))])
    frame = frame.loc[:, list(ALL_COLUMNS)]

    validate_transactions(frame, min_rows=1, require_sorted=True)
    return frame


def write_transactions_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """把模拟交易写为 CSV，必要时创建父目录。

    时间戳以秒精度 ISO 8601 文本写出，便于人工检查与跨工具读取。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = frame.copy()
    serializable[TIMESTAMP_COLUMN] = serializable[TIMESTAMP_COLUMN].dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    serializable.to_csv(path, index=False)
    return path
