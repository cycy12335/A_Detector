"""Score an external CSV through the running demo service.

The adapter deliberately owns CSV I/O and batching, while the model bundle owns
the fitted imputation, scaling, and categorical encoding.  It never sends an
optional label column to the service.

Example::

    python scripts/score_external_csv.py \
        --input data/incoming/external.csv \
        --output data/output/external_scores.csv \
        --id-column TransactionID \
        --label-column isFraud

The Docker Compose batch service invokes the same script with an internal
service URL and read-only input / writable output mounts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_BATCH_SIZE = 100
EXPECTED_FEATURE_COUNT = 199


class CsvScoringError(RuntimeError):
    """A safe, user-facing CSV adapter failure."""


def _read_json_response(
    base_url: str,
    route: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + route,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        error_code = parsed.get("error", {}).get("code", "http_error")
        raise CsvScoringError(f"服务请求失败: route={route} status={error.code} code={error_code}") from None
    except urllib.error.URLError as error:
        raise CsvScoringError(f"无法连接评分服务: route={route} reason={error.reason}") from None

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        raise CsvScoringError(f"评分服务返回了非 JSON 响应: route={route}") from None
    if not isinstance(parsed, dict):
        raise CsvScoringError(f"评分服务返回格式不正确: route={route}")
    return parsed


def load_model_contract(base_url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    model = _read_json_response(base_url, "/v1/model", timeout=timeout)
    required = model.get("required_features")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise CsvScoringError("/v1/model 未返回有效的 required_features")
    if len(required) != EXPECTED_FEATURE_COUNT or len(set(required)) != len(required):
        raise CsvScoringError(
            f"服务特征契约异常: expected={EXPECTED_FEATURE_COUNT} actual={len(required)}"
        )
    return model


def read_csv_header(path: Path, *, encoding: str, delimiter: str) -> list[str]:
    try:
        with path.open("r", encoding=encoding, newline="") as handle:
            header = next(csv.reader(handle, delimiter=delimiter), None)
    except FileNotFoundError:
        raise CsvScoringError(f"输入文件不存在: {path.name}") from None
    except UnicodeError:
        raise CsvScoringError(f"输入文件编码无法解析: encoding={encoding}") from None
    if not header:
        raise CsvScoringError("输入 CSV 没有表头")
    if any(not name.strip() for name in header):
        raise CsvScoringError("输入 CSV 存在空列名")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise CsvScoringError(f"输入 CSV 存在重复列名: {duplicates[:5]}")
    return header


def validate_header(
    header: list[str],
    required_features: list[str],
    *,
    id_column: str | None,
) -> None:
    missing = sorted(set(required_features) - set(header))
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise CsvScoringError(f"输入 CSV 缺少 {len(missing)} 个模型特征: {preview}{suffix}")
    if id_column and id_column not in header:
        raise CsvScoringError(f"输入 CSV 缺少指定 ID 列: {id_column}")


def records_from_frame(frame: pd.DataFrame, required_features: list[str]) -> list[dict[str, Any]]:
    """Convert one frame to JSON-native feature dictionaries.

    ``DataFrame.to_json`` converts pandas NaN and +/-Infinity to JSON null and
    ``json.loads`` converts numpy scalar values to ordinary Python values.
    """
    raw = frame.loc[:, required_features].to_json(
        orient="records", force_ascii=False, double_precision=15
    )
    records = json.loads(raw)
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise CsvScoringError("CSV 行无法转换为 JSON 特征记录")
    return records


def _string_id(value: Any, fallback: int) -> str:
    if value is None or pd.isna(value):
        return str(fallback)
    return str(value)


def _iter_input_chunks(
    path: Path,
    columns: list[str],
    *,
    batch_size: int,
    encoding: str,
    delimiter: str,
    max_rows: int | None,
) -> Iterable[pd.DataFrame]:
    try:
        reader = pd.read_csv(
            path,
            usecols=columns,
            chunksize=batch_size if max_rows is None else min(batch_size, max_rows),
            nrows=max_rows,
            encoding=encoding,
            sep=delimiter,
            low_memory=False,
        )
        yield from reader
    except (OSError, UnicodeError, ValueError) as error:
        raise CsvScoringError(f"读取 CSV 失败: {type(error).__name__}") from None


def score_csv(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    model = load_model_contract(args.base_url, timeout=args.timeout)
    required_features = list(model["required_features"])

    header = read_csv_header(
        input_path,
        encoding=args.encoding,
        delimiter=args.delimiter,
    )
    validate_header(header, required_features, id_column=args.id_column)

    label_column = args.label_column if args.label_column in header else None
    selected_columns = list(required_features)
    for optional_column in (args.id_column, label_column):
        if optional_column and optional_column not in selected_columns:
            selected_columns.append(optional_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    metadata_path = output_path.with_name(output_path.name + ".meta.json")
    partial_metadata_path = metadata_path.with_name(metadata_path.name + ".partial")
    fieldnames = ["row_id", "score", "flagged"]
    if label_column:
        fieldnames.append("label")

    total_rows = 0
    scored_rows = 0
    partial_path.unlink(missing_ok=True)
    partial_metadata_path.unlink(missing_ok=True)

    try:
        with partial_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for frame in _iter_input_chunks(
                input_path,
                selected_columns,
                batch_size=args.batch_size,
                encoding=args.encoding,
                delimiter=args.delimiter,
                max_rows=args.max_rows,
            ):
                features = records_from_frame(frame, required_features)
                items = [
                    {
                        "client_item_id": _string_id(
                            frame.iloc[index][args.id_column] if args.id_column else None,
                            total_rows + index,
                        ),
                        "features": feature_row,
                    }
                    for index, feature_row in enumerate(features)
                ]
                response = _read_json_response(
                    args.base_url,
                    "/v1/score/batch",
                    payload={"items": items},
                    timeout=args.timeout,
                )
                results = response.get("results")
                if not isinstance(results, list) or len(results) != len(items):
                    raise CsvScoringError("评分服务返回的结果数量与输入批次不一致")

                for index, result in enumerate(results):
                    if not isinstance(result, dict):
                        raise CsvScoringError("评分服务返回了无效的逐行结果")
                    output_row: dict[str, Any] = {
                        "row_id": items[index]["client_item_id"],
                        "score": result.get("score"),
                        "flagged": result.get("flagged"),
                    }
                    if label_column:
                        output_row["label"] = frame.iloc[index][label_column]
                    writer.writerow(output_row)

                total_rows += len(frame)
                scored_rows += len(results)

        if total_rows == 0:
            raise CsvScoringError("输入 CSV 不包含数据行")

        metadata = {
            "schema_version": "external_csv_scoring_v1",
            "input_file": input_path.name,
            "output_file": output_path.name,
            "input_rows": total_rows,
            "scored_rows": scored_rows,
            "label_column_present": bool(label_column),
            "feature_count": len(required_features),
            "model_version": model.get("model_version"),
            "bundle_sha256": model.get("bundle_sha256"),
            "threshold": model.get("threshold"),
            "threshold_policy": model.get("threshold_policy"),
        }
        partial_metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        partial_path.replace(output_path)
        partial_metadata_path.replace(metadata_path)
        return metadata
    except Exception:
        # A partial file is intentionally retained with an explicit suffix so it
        # cannot be mistaken for a complete result.
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过运行中的评分服务批量处理外部 CSV")
    parser.add_argument("--input", required=True, help="输入 CSV 路径")
    parser.add_argument("--output", required=True, help="输出评分 CSV 路径")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="评分服务地址")
    parser.add_argument("--id-column", default=None, help="结果关联 ID 列；不填则使用行号")
    parser.add_argument("--label-column", default="isFraud", help="可选本地标签列")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-rows", type=int, default=None, help="仅处理前 N 行，用于冒烟测试")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.batch_size <= DEFAULT_BATCH_SIZE:
        print("csv_scoring_failed:batch_size_must_be_1_to_100", file=sys.stderr)
        return 2
    if args.max_rows is not None and args.max_rows < 1:
        print("csv_scoring_failed:max_rows_must_be_positive", file=sys.stderr)
        return 2
    try:
        metadata = score_csv(args)
    except (CsvScoringError, OSError, ValueError) as error:
        print(f"csv_scoring_failed:{error}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
