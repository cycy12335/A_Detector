"""Run one controlled DeepSeek CW-017 synthetic end-to-end case.

The API key is read only from ``DEEPSEEK_API_KEY`` and never written to the report.
The report contains metadata only; candidate, prompt, response and decision ID are omitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.investigation import (  # noqa: E402
    ControlledGenerator,
    DeepSeekChatTransport,
    GenerationConfig,
    build_context,
    synthetic_decision,
)
from ai_detector.retrieval import build_index  # noqa: E402


def run(output: Path | None = None) -> dict[str, object]:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    index = build_index()
    policy = next(chunk for chunk in index.chunks if chunk.authority == "policy")
    snapshot = synthetic_decision("complete_low", policy_sha256=policy.source_sha256, policy_version=policy.source_version)
    context = build_context(snapshot, index)
    config = GenerationConfig(
        provider_name="deepseek",
        model_name="deepseek-flash",
        mode="real",
        credential_env="DEEPSEEK_API_KEY",
        max_cost_usd=0.07,
        # Reserve above the maximum cost of this bounded input and 2,048 output tokens.
        estimated_cost_per_call_usd=0.035,
        max_retries=1,
        max_calls=2,
        rate_limit_calls=2,
    )
    outcome = ControlledGenerator(config, DeepSeekChatTransport()).generate(snapshot, context, index)
    report = {
        "schema_version": "cw017_deepseek_e2e_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "deepseek",
        "model": "deepseek-flash",
        "prompt_version": config.prompt_version,
        "sample_count": 1,
        "real_e2e_evidence": True,
        "status": outcome.status,
        "validation_status": outcome.validation_status,
        "error_codes": list(outcome.error_codes),
        "attempts": outcome.attempts,
        "calls": outcome.calls,
        "request_chars": outcome.request_chars,
        "response_chars": outcome.response_chars,
        "cost_usd": outcome.cost_usd,
        "budget_usd": config.max_cost_usd,
        "raw_payloads_saved": False,
        "credentials_saved": False,
        "decision_identifiers_saved": False,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one bounded DeepSeek CW-017 synthetic case.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/cw017_deepseek_e2e/report.json"))
    args = parser.parse_args()
    try:
        report = run(args.output if not args.output.is_absolute() else args.output)
    except (OSError, RuntimeError, ValueError, KeyError, StopIteration) as error:
        print(f"deepseek_e2e_failed:{type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"verified", "insufficient_evidence"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
